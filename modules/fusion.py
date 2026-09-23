
import torch
import torch.nn as nn
from modules.model import WanLayerNorm, WanModel, WanRMSNorm, gradient_checkpointing, rope_apply
from modules.attention import flash_attention
from distributed_comms.communications import all_gather, all_to_all_4D
from distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state


class FusionModel(nn.Module):
    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        has_video = True 
        has_audio = True
        if video_config is not None:
            self.video_model = WanModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")
        
        if audio_config is not None:
            self.audio_model = WanModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)

            self.use_sp = get_sequence_parallel_state()
            if self.use_sp:
                self.sp_size = nccl_info.sp_size
                self.sp_rank = nccl_info.rank_within_group
            self.inject_cross_attention_kv_projections()

        self.init_weights()

        self.gradient_checkpointing = False

        # reference conditioning settings, overridden from the run config by the pipeline
        self.n_refs = 1
        self.use_ref_av_fusion = False

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    def init_lora(self, self_lora=False, train=True, vid_ip_emb_dim=None, audio_ip_emb_dim=None,
                  use_ref_av_fusion=None, fusion_lora_rank=128):
        if use_ref_av_fusion is not None:
            self.use_ref_av_fusion = use_ref_av_fusion
        fusion_lora = self.use_ref_av_fusion
        if self.video_model is not None:
            self.video_model.init_lora(self_lora, train, vid_ip_emb_dim,
                                       fusion_lora=fusion_lora, fusion_lora_rank=fusion_lora_rank)
        if self.audio_model is not None:
            self.audio_model.init_lora(self_lora, train, audio_ip_emb_dim,
                                       fusion_lora=fusion_lora, fusion_lora_rank=fusion_lora_rank)

    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
        
        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()

    @staticmethod
    def fusion_kv(cross_attn_block, target_seq):
        """Keys/values of the fusion cross-attention for a target sequence.

        Shared by the main (audio<->video) route and the reference-pair route, so both see the
        same projections and the optional fusion LoRA.
        """
        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k = cross_attn_block.k_fusion(target_seq)
        v = cross_attn_block.v_fusion(target_seq)
        # `k_fusion_lora`/`v_fusion_lora` are added by `cross_attn.init_fusion_lora`; their `up`
        # projection is zero initialized, so this is a no-op until they are trained.
        if hasattr(cross_attn_block, "k_fusion_lora"):
            k = k + cross_attn_block.k_fusion_lora(target_seq)
            v = v + cross_attn_block.v_fusion_lora(target_seq)
        k = cross_attn_block.norm_k_fusion(k)
        return k, v

    def ref_pair_cross_attention(self,
                                 cross_attn_block,
                                 q_seq,
                                 q_grid_sizes,
                                 q_freqs,
                                 q_offsets,
                                 kv_seq,
                                 kv_grid_sizes,
                                 kv_freqs,
                                 kv_offsets,
                                 n_refs=1
                                 ):
        """Cross-modal attention between the reference tokens of the *same* person.

        The main fusion cross-attention links the two main sequences; reference tokens do not take
        part in it. This pairs slot j of one modality with slot j of the other, so a reference face
        and the matching reference voice exchange information - that is what pins "this appearance"
        to "this timbre". Pairing is block diagonal, i.e. no information crosses between persons.
        """
        b, n, d = q_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        assert n_refs >= 1 and q_seq.size(1) % n_refs == 0 and kv_seq.size(1) % n_refs == 0, \
            f"Reference tokens must split evenly into {n_refs} persons, got {q_seq.size(1)} and {kv_seq.size(1)}"
        per_q, per_kv = q_seq.size(1) // n_refs, kv_seq.size(1) // n_refs

        q = cross_attn_block.norm_q(cross_attn_block.q(q_seq)).view(b, -1, n, d)
        k, v = self.fusion_kv(cross_attn_block, kv_seq)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        q = rope_apply(q, q_grid_sizes, q_freqs, q_offsets)
        k = rope_apply(k, kv_grid_sizes, kv_freqs, kv_offsets)

        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)

        paired = []
        for j in range(n_refs):
            paired.append(flash_attention(
                q[:, j * per_q:(j + 1) * per_q],
                k[:, j * per_kv:(j + 1) * per_kv],
                v[:, j * per_kv:(j + 1) * per_kv],
                k_lens=None))
        x = torch.cat(paired, dim=1)

        if self.use_sp:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        x = x.flatten(2)
        return cross_attn_block.o(x)

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def single_fusion_cross_attention_forward(self,
                                            cross_attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens
                                            ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            ## means is i2v block
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            ## means is t2v block
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None
        
        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            if k_img is not None:
                k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            if v_img is not None:
                v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]
            
        x = flash_attention(q, k, v, k_lens=context_lens)

        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x

        is_vid = src_grid_sizes.shape[1] > 1
        # compute target attention
        k_target, v_target = self.fusion_kv(cross_attn_block, target_seq)
        k_target = k_target.view(b, -1, n, d)
        v_target = v_target.view(b, -1, n, d)
        if self.use_sp: 
            k_target = all_to_all_4D(k_target, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
            v_target = all_to_all_4D(v_target, scatter_dim=2, gather_dim=1) # [B, L, H/P, C/H]
        
        q = rope_apply(q, src_grid_sizes, src_freqs)
        k_target = rope_apply(k_target, target_grid_sizes, target_freqs)
        
        target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)
        
        x = x + target_x
        if self.use_sp:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2) # [B, L/P, H, C/H]
        
        x = x.flatten(2) # [B, L/P, C]

        x = cross_attn_block.o(x)
        return x

    def single_fusion_cross_attention_ffn_forward(self,
                                            attn_block,
                                            src_seq,
                                            src_grid_sizes,
                                            src_freqs,
                                            target_seq,
                                            target_seq_lens,
                                            target_grid_sizes,
                                            target_freqs,
                                            context,
                                            context_lens,
                                            src_e):
        
        src_seq = src_seq + self.single_fusion_cross_attention_forward(attn_block.cross_attn,
                                                                       attn_block.norm3(src_seq),
                                                                       src_grid_sizes=src_grid_sizes,
                                                                       src_freqs=src_freqs,
                                                                       target_seq=target_seq,
                                                                       target_seq_lens=target_seq_lens,
                                                                       target_grid_sizes=target_grid_sizes,
                                                                       target_freqs=target_freqs,
                                                                       context=context,
                                                                       context_lens=context_lens
                                                                       )
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq
        
    def single_fusion_block_forward(self,
                                    vid_block,
                                    audio_block,
                                    vid,
                                    audio,
                                    vid_ip,
                                    audio_ip,
                                    vid_e,
                                    vid_seq_lens,
                                    vid_grid_sizes,
                                    vid_freqs,
                                    vid_context,
                                    vid_context_lens,
                                    vid_e_ip,
                                    vid_ip_grid_sizes,
                                    vid_ip_freqs,
                                    vid_ip_offsets,
                                    vid_ip_emb,
                                    audio_e,
                                    audio_seq_lens,
                                    audio_grid_sizes,
                                    audio_freqs,
                                    audio_context,
                                    audio_context_lens,
                                    audio_e_ip,
                                    audio_ip_grid_sizes,
                                    audio_ip_freqs,
                                    audio_ip_offsets,
                                    audio_ip_emb,
                                    n_refs=1,
                                    use_ref_av_fusion=False
                                    ):
        ## audio modulation
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6 and audio_e.shape[1] == audio.shape[1], f"{audio_e.shape}, {audio.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_e = audio_block.modulation(audio_e).chunk(6, dim=2)
        assert audio_e[0].dtype == torch.bfloat16

        ## audio_ip modulation
        if audio_e_ip is not None \
            and audio_ip is not None \
            and audio_ip_grid_sizes is not None \
            and audio_ip_freqs is not None:
            assert audio_e_ip.dtype == torch.bfloat16
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                audio_e_ip = audio_block.modulation(audio_e_ip).chunk(6, dim=2)
            assert audio_e_ip[0].dtype == torch.bfloat16

            input_audio_ip = audio_block.norm1(audio_ip).bfloat16() * (1 + audio_e_ip[1].squeeze(2)) + audio_e_ip[0].squeeze(2)
        else:
            input_audio_ip = None
        # audio self-attention
        audio_y, audio_y_ip = audio_block.self_attn(
            audio_block.norm1(audio).bfloat16() * (1 + audio_e[1].squeeze(2)) + audio_e[0].squeeze(2), 
            audio_seq_lens, audio_grid_sizes, audio_freqs,
            input_audio_ip, audio_ip_grid_sizes, audio_ip_freqs, 
            audio_ip_offsets, audio_ip_emb
        )
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio = audio + audio_y * audio_e[2].squeeze(2)
            if audio_ip is not None:
                audio_ip = audio_ip + audio_y_ip * audio_e_ip[2].squeeze(2)

        ## video modulation
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6 and vid_e.shape[1] == vid.shape[1], f"{vid_e.shape}, {vid.shape}"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid_e = vid_block.modulation(vid_e).chunk(6, dim=2)
        
        if vid_e_ip is not None \
            and vid_ip is not None \
            and vid_ip_grid_sizes is not None \
            and vid_ip_freqs is not None:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vid_e_ip = vid_block.modulation(vid_e_ip).chunk(6, dim=2)
            input_vid_ip = vid_block.norm1(vid_ip).bfloat16() * (1 + vid_e_ip[1].squeeze(2)) + vid_e_ip[0].squeeze(2)
        else:
            input_vid_ip = None

        # video self-attention
        vid_y, vid_y_ip = vid_block.self_attn(
            vid_block.norm1(vid).bfloat16() * (1 + vid_e[1].squeeze(2)) + vid_e[0].squeeze(2), 
            vid_seq_lens, vid_grid_sizes, vid_freqs,
            input_vid_ip, vid_ip_grid_sizes, vid_ip_freqs, 
            vid_ip_offsets, vid_ip_emb
        )

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vid = vid + vid_y * vid_e[2].squeeze(2)
            if vid_ip is not None:
                vid_ip = vid_ip + vid_y_ip * vid_e_ip[2].squeeze(2)

        og_audio = audio

        # audio cross-attention
        audio = self.single_fusion_cross_attention_ffn_forward(
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self.single_fusion_cross_attention_ffn_forward(
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e
        )

        # reference-pair audio<->video fusion: person j's reference face tokens attend to person
        # j's reference audio tokens and vice versa, which pins a face to its voice. Reference
        # tokens are excluded from the main fusion cross-attention above, so this is additive.
        if use_ref_av_fusion and vid_ip is not None and audio_ip is not None \
            and vid_ip_grid_sizes is not None and audio_ip_grid_sizes is not None:
            vid_pair = self.ref_pair_cross_attention(
                vid_block.cross_attn,
                vid_ip, vid_ip_grid_sizes, vid_ip_freqs, vid_ip_offsets,
                audio_ip, audio_ip_grid_sizes, audio_ip_freqs, audio_ip_offsets,
                n_refs=n_refs)
            audio_pair = self.ref_pair_cross_attention(
                audio_block.cross_attn,
                audio_ip, audio_ip_grid_sizes, audio_ip_freqs, audio_ip_offsets,
                vid_ip, vid_ip_grid_sizes, vid_ip_freqs, vid_ip_offsets,
                n_refs=n_refs)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vid_ip = vid_ip + vid_pair
                audio_ip = audio_ip + audio_pair

        # process vid_ip and audio_ip if needed
        if audio_ip is not None:
            audio_y_ip = vid_block.ffn(
                audio_block.norm2(audio_ip).bfloat16() * (1 + audio_e_ip[4].squeeze(2)) + audio_e_ip[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                audio_ip = audio_ip + audio_y_ip * audio_e_ip[5].squeeze(2)
        
        if vid_ip is not None:
            vid_y_ip = vid_block.ffn(
                vid_block.norm2(vid_ip).bfloat16() * (1 + vid_e_ip[4].squeeze(2)) + vid_e_ip[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                vid_ip = vid_ip + vid_y_ip * vid_e_ip[5].squeeze(2)

        return vid, vid_ip, audio, audio_ip

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
        vid_ip=None,
        audio_ip=None,
        vid_ip_emb=None,
        audio_ip_emb=None
    ):  

        assert clip_fea is None 
        assert y is None

        if vid is None or all([x is None for x in vid]):
            assert vid_context is None
            assert vid_seq_len is None
            assert self.audio_model is not None

            return None, self.audio_model(x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None)
        
        if audio is None or all([x is None for x in audio]):
            assert clip_fea_audio is None
            assert audio_context is None
            assert audio_seq_len is None
            assert self.video_model is not None

            return self.video_model(x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, first_frame_is_clean=first_frame_is_clean), None
        
        vid, vid_e, vid_ip, vid_e_ip, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, clip_fea=clip_fea, y=y, 
            first_frame_is_clean=first_frame_is_clean, x_ip=vid_ip, ip_emb=vid_ip_emb
        )

        audio, audio_e, audio_ip, audio_e_ip, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio, t=t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None, 
            first_frame_is_clean=False, x_ip=audio_ip, ip_emb=audio_ip_emb
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            vid, vid_ip, audio, audio_ip = gradient_checkpointing(
                    enabled=self.gradient_checkpointing,
                    module=self.single_fusion_block_forward,
                    vid_block=vid_block,
                    audio_block=audio_block,
                    vid=vid,
                    audio=audio,
                    vid_ip=vid_ip,
                    audio_ip=audio_ip,
                    n_refs=self.n_refs,
                    use_ref_av_fusion=self.use_ref_av_fusion,
                    **kwargs
                )

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs['grid_sizes'], vid_e)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs['grid_sizes'], audio_e)
        # TODO: process vid_ip and audio_ip if needed

        return vid, audio

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    
    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()