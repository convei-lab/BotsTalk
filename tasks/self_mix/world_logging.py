#!/usr/bin/env python3

# Copyright (c) Conversation Intelligence Lab and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful utilities for logging actions/observations in a world.
"""

from random import random
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.worlds import BatchWorld, DynamicBatchWorld
from parlai.utils.misc import msg_to_str
from parlai.utils.conversations import Conversations
from parlai.utils.io import PathManager
import parlai.utils.logging as logging
from parlai.core.message import Message
from parlai.utils.world_logging import WorldLogger

import os
import copy
from tqdm import tqdm
import json
import random
KEEP_ALL = 'all'

class DebateLogger(WorldLogger):

    def _add_msgs(self, acts, idx=0):
        """
        Add messages from a `parley()` to the current episode of logs.

        :param acts: list of acts from a `.parley()` call
        """
        msgs = []
        for act_pair in acts:
            msg_pair = []
            for act in act_pair:
                # padding examples in the episode[0]
                if not isinstance(act, Message):
                    act = Message(act)
                if act.is_padding():
                    break
                if not self.keep_all:
                    msg = {f: act[f] for f in self.keep_fields if f in act}
                else:
                    msg = act
                msg_pair.append(msg)
            msgs.append(msg_pair)

        if len(msgs) == 0:
            return
        self._current_episodes.setdefault(idx, [])
        self._current_episodes[idx].append(msgs)

    def convert_to_labeled_data(self, episode, opt, responses):
        """
        Write one episode into series of parl-ai format lines
        """
        debug = False
        out = []
        
        # dialogue initiation
        subtasks = opt['subtasks']
        text_lst = []
        partner_context = []
        subtask_context = {}
        seeded = False
        ntasks = len(subtasks)
        assert len(episode) % ntasks == 0
        
        for i, expertise in enumerate(episode):
            line = {'id': '', 'text': '', 'labels': '', 'episode_done': False, 'first_expertise': '', 'second_expertise': ''}

            for j, (subtask, parley) in enumerate(zip(subtasks, expertise)):
                first_act, second_act = parley
                if first_act['id'] == 'context' and second_act['id'] == 'context': # context
                    text_lst.append(second_act['text'])
                    partner_context.append(first_act['text'])
                    subtask_context[f'first_context_{subtask}'] = first_act['text']
                    subtask_context[f'second_context_{subtask}'] = second_act['text']
                elif first_act['id'] == 'seed' and second_act['id'] == 'seed': # seed
                    if not seeded: # Writing first seed
                        text_lst.append(first_act['text'])

                        line['id'] = 'context'
                        line['text'] = '\n'.join(text_lst)
                        line['labels'] = [second_act['text']]
                        line['partner_context'] = '\n'.join(partner_context) if partner_context else ''
                        line['label_candidates'] = random.sample(responses, 100)
                        line['label_candidates'][random.randrange(100)] = line['labels'][0]
                        for k, v in subtask_context.items(): line[k] = v
                        out.append(line)
                        if debug: print('*line*', line, '\n'); input()
                        text_lst = []
                        seeded = True
                else: # utterance
                    first_verdict = first_act['verdict'].split(',')
                    second_verdict = second_act['verdict'].split(',')
                    first_decision = first_act['decision'].split(',')
                    second_decision = second_act['decision'].split(',')
                    if '1' in first_decision:
                        line['id'] = first_act['id']
                        text_lst.append(first_act['text'])
                        line['first_expertise'] = subtasks[j]
                    if '1' in second_decision:
                        line['labels'] = [second_act['text']]
                        line['second_expertise'] = subtasks[j]
                    for k, (text, score) in enumerate(first_act['beam_texts']):
                        line[f'first_{subtask}_{k}'] = text + f' (score: {str(round(score, 2))}, verdict: {first_verdict[k]}, decision: {first_decision[k]})'
                    for k, (text, score) in enumerate(second_act['beam_texts']):
                        line[f'second_{subtask}_{k}'] = text + f' (score: {str(round(score, 2))}, verdict: {second_verdict[k]}, decision: {second_decision[k]})' 
                if debug: print(first_act); print(second_act); input()
            # In case of utterances, we collect the best while iteration.
            if second_act['id'] != 'context' and second_act['id'] != 'seed':
                line['text'] = '\n'.join(text_lst)
                line['label_candidates'] = random.sample(responses, 100)
                line['label_candidates'][random.randrange(100)] = line['labels'][0]
                out.append(line)
                if debug: print('*line*', line, '\n'); input()
                text_lst = []
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def convert_to_analytical_data(self, episode, opt):
        """
        Write one episode into series of parl-ai format lines
        """
        debug = False
        out = []
        
        # dialogue initiation
        subtasks = opt['subtasks']
        partner_context = []
        subtask_context = {}
        seeded = False
        ntasks = len(subtasks)
        assert len(episode) % ntasks == 0
        
        for i, expertise in enumerate(episode):
            line = {'id': '', 'text': '', 'labels': '', 'episode_done': False, 'first_expertise': '', 'second_expertise': ''}
            text_lst = [] # By reinitializing text fields, we're changing 
                          # the first utterance pairs to exclude context information

            for j, (subtask, parley) in enumerate(zip(subtasks, expertise)):
                first_act, second_act = parley
                if first_act['id'] == 'context' and second_act['id'] == 'context': # context
                    text_lst.append(second_act['text'])
                    partner_context.append(first_act['text'])
                    subtask_context[f'first_context_{subtask}'] = first_act['text']
                    subtask_context[f'second_context_{subtask}'] = second_act['text']
                elif first_act['id'] == 'seed' and second_act['id'] == 'seed': # seed
                    if not seeded: # Writing first seed
                        text_lst.append(first_act['text'])

                        line['id'] = 'context'
                        line['text'] = '\n'.join(text_lst)
                        line['labels'] = [second_act['text']]
                        line['partner_context'] = '\n'.join(partner_context) if partner_context else ''
                        for k, v in subtask_context.items(): line[k] = v
                        out.append(line)
                        if debug: print('*line*', line, '\n'); input()
                        text_lst = []
                        seeded = True
                else: # utterance
                    first_verdict = first_act['verdict'].split(',')
                    second_verdict = second_act['verdict'].split(',')
                    first_decision = first_act['decision'].split(',')
                    second_decision = second_act['decision'].split(',')
                    if '1' in first_decision:
                        line['id'] = first_act['id']
                        text_lst.append(first_act['text'])
                        line['first_expertise'] = subtasks[j]
                    if '1' in second_decision:
                        line['labels'] = [second_act['text']]
                        line['second_expertise'] = subtasks[j]
                    for k, (text, score) in enumerate(first_act['beam_texts']):
                        line[f'first_{subtask}_{k}'] = text + f' (score: {str(round(score, 2))}, verdict: {first_verdict[k]}, decision: {first_decision[k]})'
                    for k, (text, score) in enumerate(second_act['beam_texts']):
                        line[f'second_{subtask}_{k}'] = text + f' (score: {str(round(score, 2))}, verdict: {second_verdict[k]}, decision: {second_decision[k]})'  
                if debug: print(first_act); print(second_act); input()
            # We register the cached text & labels from the above task-wise iteration, colleted as dialogue histories.
            if second_act['id'] != 'context' and second_act['id'] != 'seed':
                line['text'] = '\n'.join(text_lst)
                out.append(line)
                if debug: print('*line*', line, '\n'); input()
                text_lst = []
        if len(out) > 0:
            out[-1]['episode_done'] = True
        return out

    def write(self, outfile, world, opt, indent=4):
        file_format = opt['save_format'] 
        if file_format == 'conversations':
            self.write_conversations_format(outfile, world)
        else:
            # ParlAI text format
            self.write_parlai_format(outfile, opt)

    def write_parlai_format(self, outfile, opt):
        logging.info(f'Saving log to {outfile} in ParlAI format')
        
        res_path = os.path.join(opt['datapath'], 'pbst', 'responses_candidates.json')
        with PathManager.open(res_path, 'r') as file:
            responses = json.load(file)
            
        # For debug
        dt = opt['datatype'].split(':')[0]
        # ana_path = os.path.join(opt['datapath'], 'pbst', f'machine_analysis_{dt}.json')
        ana_path = opt['outfile'][:-4] + '.json'
        logging.info(f'Saving analytical dataset to {ana_path}')
            
        with PathManager.open(outfile, 'w') as fw, PathManager.open(ana_path, 'w') as afw:
            for episode in tqdm(self._logs):
                ep = self.convert_to_labeled_data(episode, opt, responses)
                for act in ep:
                    txt = msg_to_str(act)
                    fw.write(txt + '\n')
                fw.write('\n')
                
                ep = self.convert_to_analytical_data(episode, opt)
                for act in ep:
                    txt = msg_to_str(act)
                    for t in txt.split('\t'):
                        if t.startswith('id:'):
                            afw.write('\n')
                        afw.write('-> '+ t + '\n')
                afw.write('\n')

    # TODO this need updates
    def write_conversations_format(self, outfile, world):
        logging.info(f'Saving log to {outfile} in Conversations format')
        Conversations.save_conversations(
            self._logs,
            outfile,
            world.opt,
            self_chat=world.opt.get('selfchat_task', False),
        )