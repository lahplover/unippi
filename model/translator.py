import torch
from .transformer import Transformer
from .beam import Beam

''' This module will handle the text generation with beam search. '''


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if args.cuda else 'cpu')
        self.seq_len = args.seq_len
        self.n_best = args.n_best
        self.beam_size = args.beam_size

        # checkpoint = torch.load(args.model)
        # model_opt = checkpoint['settings']
        # self.model_opt = model_opt

        model = Transformer(args.vocab_len, hidden=args.hidden, n_layers=args.layers,
                            attn_heads=args.attn_heads, seq_length=args.seq_len)

        if args.cuda:
            model.load_state_dict(torch.load(args.model))
        else:
            model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

        print('[Info] Trained model state loaded.')

        # model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def translate_batch(self, src_seq):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(memory, memory_mask, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_memory_mask = collect_active_part(memory_mask, active_inst_idx, n_prev_active_inst, n_bm)
            active_memory = collect_active_part(memory, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_memory, active_memory_mask, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, memory, memory_mask, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            # def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            #     dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
            #     dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            #     return dec_partial_pos

            def predict_word(dec_seq, memory, memory_mask, n_active_inst, n_bm):
                dec_output = self.model.decoder(dec_seq, memory, memory_mask)
                # print(dec_output.size())
                dec_output = dec_output[-1:, :, :]  # Pick the last step: (bh * bm) * d_h
                # word_prob = F.log_softmax(self.model.MaskedLanguageModel(dec_output), dim=1)
                word_prob = self.model.mask_lm(dec_output)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                # print(word_prob.size())
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            # dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, memory, memory_mask, n_active_inst, n_bm)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # Encode
            src_seq = src_seq.to(self.device)
            memory, memory_mask = self.model.encoder(src_seq)

            # Repeat data for beam search
            n_bm = self.beam_size
            len_s, n_inst, d_h = memory.size()
            memory = memory.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            memory_mask = memory_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)

            # Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # Decode
            for len_dec_seq in range(1, self.seq_len + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, memory, memory_mask, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                memory, memory_mask, inst_idx_to_position_map = collate_active_info(
                    memory, memory_mask, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.n_best)

        return batch_hyp, batch_scores
