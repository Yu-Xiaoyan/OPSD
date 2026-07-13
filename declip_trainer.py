"""OPSDDeclipTrainer (de-clip 第1步 C/D 臂) — 逐样本腐蚀门蒸馏。

基底 = repo 口径但 jsd_token_clip=0（无 clip）。在 wrong+correct rollout 上：
  1. student 前向（带 grad）+ correct-teacher 前向（no_grad, base）+ corrupt-teacher 前向
     （no_grad, base；corrupt_solution 把 gt 答案替换成无关腐蚀答案）；
  2. 逐 token 腐蚀敏感度 c_t = token_jsd(T_S, T_S̃)；阈 ρ（诊断集 90 分位，硬编、不调参）；
  3. arm C（硬门）：敏感位 c_t>ρ 直接扔掉（不入 loss），其余按 forward-KL(T_S‖S) 蒸馏；
     arm D（边缘化）：敏感位目标换 P̄_T=0.5[P_T+P_T̃]（logits 层 log-mean-exp、renorm），其余=T_S。
禁止物化多余全词表中间量（流式：单样本 [1,T,V]，用完即 del）。
"""
from __future__ import annotations
import math, os, sys
import torch
import torch.nn.functional as F
from opsd_trainer import OPSDTrainer

_PROBES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes")
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
from scoring import (build_teacher_prompt_text, build_student_prompt_text,  # noqa: E402
                     forward_rollout_logits, teacher_mode)
from divergence import token_jsd  # noqa: E402
from corrupt_solution import corrupt_solution  # noqa: E402
from corrupt_answers import corrupt_answer  # noqa: E402

DECLIP_RHO = 0.0007  # diag2x2_shard0 的 c_t 90 分位（出处见 docs/v1_design.md），不调参


class OPSDDeclipTrainer(OPSDTrainer):
    def __init__(self, *args, declip_arm="C", declip_rho=DECLIP_RHO, **kwargs):
        super().__init__(*args, **kwargs)
        assert declip_arm in ("C", "D"), declip_arm
        self.declip_arm = declip_arm
        self.declip_rho = float(declip_rho)
        print(f"[OPSDDeclipTrainer] arm={self.declip_arm} rho={self.declip_rho}")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._signature_columns is not None:
            for c in ("Answer", "solution"):
                if c not in self._signature_columns:
                    self._signature_columns.append(c)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_thinking_machines_loss:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        tok = self.processing_class
        sp_len = inputs["student_prompt_length"]
        sampled = inputs["student_input_ids"][:, sp_len:]
        shifted = inputs["labels"][:, sp_len:]
        mask = shifted != -100
        problems = inputs.get("problems"); gts = inputs.get("gt_answers"); sols = inputs.get("solutions")
        dev = inputs["student_input_ids"].device
        B = sampled.shape[0]
        total = torch.zeros((), device=dev); ntok = 0
        n_sens = 0; n_all = 0; probe_ms = 0.0
        for i in range(B):
            ids = sampled[i][mask[i]].tolist()
            if len(ids) < 4:
                continue
            problem = problems[i] if problems else ""
            gt = str(gts[i]) if gts else ""
            sol = sols[i] if sols else ""
            if not sol or not gt:
                continue
            # student 前向（带 grad）
            sp = build_student_prompt_text(tok, problem, student_thinking=False)
            sp_ids = tok(sp, return_tensors="pt").input_ids[0].tolist()
            s_in = torch.tensor([sp_ids + ids], device=dev)
            S = model(input_ids=s_in, attention_mask=torch.ones_like(s_in)).logits[:, len(sp_ids) - 1:-1, :].float().squeeze(0)
            # teacher 前向（no_grad, base）: correct + corrupt
            with torch.no_grad(), teacher_mode(model):
                TS, _, _ = forward_rollout_logits(model, tok, build_teacher_prompt_text(tok, problem, sol), ids)
                corr = corrupt_answer(gt, n=1, seed=i)
                if corr:
                    csol, _ = corrupt_solution(sol, gt, corr[0])
                    TSt, _, _ = forward_rollout_logits(model, tok, build_teacher_prompt_text(tok, problem, csol), ids)
                else:
                    TSt = TS
            T = min(S.shape[0], TS.shape[0], TSt.shape[0])
            S = S[:T]; TS = TS[:T]; TSt = TSt[:T]
            c_t = token_jsd(TS, TSt)[:T].to(dev)          # [T] 腐蚀敏感度
            sens = c_t > self.declip_rho
            n_sens += int(sens.sum()); n_all += T
            logS = F.log_softmax(S, dim=-1)
            logTS = F.log_softmax(TS, dim=-1)
            if self.declip_arm == "C":
                keep = ~sens
                if int(keep.sum()) == 0:
                    del S, TS, TSt; continue
                kl = (logTS.exp() * (logTS - logS)).sum(-1)   # forward KL(T_S‖S) [T]
                loss_i = kl[keep].mean(); nt = int(keep.sum())
            else:  # D: 敏感位用边缘化目标
                logTSt = F.log_softmax(TSt, dim=-1)
                log_marg = torch.logsumexp(torch.stack([logTS, logTSt], 0), 0) + math.log(0.5)  # log 0.5(P_T+P_T̃)
                logT = torch.where(sens.unsqueeze(-1), log_marg, logTS)   # 非敏感位=T_S
                kl = (logT.exp() * (logT - logS)).sum(-1)
                loss_i = kl.mean(); nt = T
                del logTSt, log_marg, logT
            total = total + loss_i * nt; ntok += nt
            del S, TS, TSt, logS, logTS, c_t
        loss = total / max(1, ntok)
        mode = "train" if model.training else "eval"
        self._metrics[mode]["declip/sens_frac"].append(n_sens / max(1, n_all))
        self._metrics[mode]["declip/n_examples"].append(float(B))
        torch.cuda.empty_cache()
        if return_outputs:
            class _O:
                def __init__(s, l): s.loss = l
            return (loss, _O(loss))
        return loss
