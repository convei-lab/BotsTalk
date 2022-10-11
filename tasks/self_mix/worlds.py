#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from typing import Any, Dict, List, Optional

from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, validate
from parlai.core.message import Message
import torch
import numpy as np

from parlai.scripts.eval_model import eval_model
from parlai.core.agents import create_agent, create_agent_from_model_file
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric, Dataset
import json
from scipy.stats import entropy 

from parlai.core.worlds import World
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tasks.pbst.agents import SelfmixTeacher
from parlai.core.worlds import BatchWorld, DialogPartnerWorld

ROBERTA = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').cuda()

def load_openers(opt) -> Optional[List[str]]:
    base_task = opt['task'].split(':')[0]
    if base_task == 'self_mix':
        return None

    print('[ loading conversation openers... ]')
    # create dummy task so we can get openers from the data
    task_opt = copy.deepcopy(opt)
    task_opt['task'] = base_task + ':SelfmixTeacher'

    # default train will loop forever, but evalmode will stop after one epoch
    datatype = task_opt['datatype']
    print('datatype of pbst task', datatype)
    if 'train' in datatype and 'evalmode' not in datatype:
        task_opt['datatype'] = f'{datatype}:evalmode'
    print(f"[ loading openers in datatype {task_opt['datatype']}")
    task_opt['interactive_task'] = False
    task_opt['selfchat_task'] = False
    task_opt['selfmix_task'] = False
    task_opt['fixed_response'] = None
    task_opt['seed_messages_from_task'] = False
    task_agent = FixedResponseAgent(task_opt)
    # task_world = create_task(task_opt, task_agent)
    task_teacher = SelfmixTeacher(task_opt)
    task_world = DialogPartnerWorld(opt=task_opt, agents=[task_teacher, task_agent])

    # run through task data, collecting all first messages
    # openers = set()
    openers = []
    source_list = []
    is_first_turn = True
    while not task_world.epoch_done():
        task_world.parley()
        msg = task_world.get_acts()[0]
        # add only the first message in the episode
        if is_first_turn and msg.get('text') and msg.get('eval_labels'):
            openers.append([msg['text'], msg['eval_labels'][0]])
            source_list.append(msg.get('source_task'))
        is_first_turn = msg.get('episode_done', False)
    # for opener in openers:
    #     print('opener', opener)


    # set seed range for pararell training
    if opt['seed_range'] != None:
        seed_start, seed_end = opt['seed_range'].split(',')
        openers = openers[int(seed_start): int(seed_end)]
        source_list = source_list[int(seed_start): int(seed_end)]

    print(f'[ loaded {len(openers)} openers ]')
    assert len(openers) == len(source_list)
    return openers, source_list


def load_openers_from_file(filepath: str) -> List[str]:
    openers = []
    with open(filepath, 'r') as f:
        openers = [l.strip() for l in f]
    return openers

class TeamDebateWorld(World):
    """
    Simple world for two group of agents communicating synchronously.

    This basic world switches back and forth between two group of agents, 
    giving each agent in a team side one chance to speak per turn and 
    passing that back to the other one.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Return the parser as-is.

        Team-debate-specific world flags can be added here.
        """
        return parser
    
    def __init__(self, opt: Opt, agents=None, shared=None):
        if not ((agents is not None) ^ (shared is not None)):
            raise ValueError('You must supply either agents or shared, but not both.')
        super_class = super()
        super_class.__init__(opt=opt)
        # super().__init__(opt=opt)
        if shared:
            # Create agents based on shared data.
            self.agents = create_agents_from_shared(shared['agents'])
        else:
            if not isinstance(agents[0], list): # one-dimensional list
                print('Warning! There must not be exactly two agents for this world.')
                if len(agents) == 2:
                    agents = [agents]
                    print('.. Repacking the given task-model agents into the 2D-list.')
                else:
                    raise RuntimeError('Error')


            for agent_pair in agents: # multi-dimensional list
                if len(agent_pair) != 2:
                    raise RuntimeError('There must be exactly two agents for each and every task.')

            # Add passed in agents directly.
            self.agents = agents
        self.acts = [[None] * 2 for _ in range(len(self.agents))]
        if self.agents is not None and len(self.agents) > 0:
            # Name the world after the first agent.
            self.id = self.get_task_agent()[0].getID()

    def get_task_agent(self):
        """
        Return task agent.
        """
        return [agent[0] for agent in self.get_agents()]

    def get_model_agent(self):
        """
        Return model agent, if applicable.
        """
        return [agent[1] for agent in self.get_agents()]

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        raise 'Function unemplemented'
        acts = self.acts
        agents = self.agents
        acts[0] = agents[0].act()
        agents[1].observe(validate(acts[0]))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()

    def episode_done(self):
        """
        Only the first agent indicates when the episode is done.
        """
        if self.acts[0][0] is not None:
            return self.acts[0][0].get('episode_done', False)
        else:
            return False

    def epoch_done(self):
        """
        Only the first agent indicates when the epoch is done.
        """
        return self.get_task_agent()[0].epoch_done()

    def report(self):
        """
        Report all metrics of all subagents.
        """
        from parlai.core.metrics import Metric, LegacyMetric

        metrics = {}
        for a in self.agents:
            if hasattr(a, 'report'):
                m = a.report()
                for k, v in m.items():
                    if not isinstance(v, Metric):
                        v = LegacyMetric(v)
                    if k not in metrics:
                        # first agent gets priority in settings values for keys
                        # this way model can't e.g. override accuracy to 100%
                        metrics[k] = v
        if metrics and 'exs' in metrics:
            self.total_exs += metrics['exs'].value()
        return metrics

    def num_examples(self):
        """
        Return number of examples.
        """
        if hasattr(self.get_task_agent(), 'num_examples'):
            return self.get_task_agent().num_examples()
        return 0

    def num_episodes(self):
        """
        Return number of episodes.
        """
        if hasattr(self.get_task_agent(), 'num_episodes'):
            return self.get_task_agent().num_episodes()
        return 0

    def shutdown(self):
        """
        Shutdown each agent.
        """
        for a in self.agents:
            a.shutdown()

    def update_counters(self):
        """
        Ensure all counters are synchronized across threads.
        """
        super().update_counters()
        for a in self.agents:
            if hasattr(a, 'update_counters'):
                a.update_counters()


class SelfMixWorld(TeamDebateWorld):
    def __init__(self, opt, agents, shared=None):
        self.agents, self.retrieval_experts = agents
        super().__init__(opt=opt, agents=self.agents, shared=shared)
        self.init_contexts(shared=shared)
        self._openers = None
        self.init_openers()
        self.max_turn_cnt = self.opt.get('selfmix_max_turns', 10)
        self.turn_cnt = 0
        self.episode_cnt = 0
        self.nsubtask = len(self.opt.get('subtasks'))
        self.world_name = 'SelfMixWorld'
        ROBERTA.eval()
        self.init_skill_classifier()
        self.active_flags = [1, 0, 0]
        self.task_to_index = {self.opt.get('subtasks')[0]:0, self.opt.get('subtasks')[1]:1, self.opt.get('subtasks')[2]:2}

    def init_contexts(self, shared=None) -> None:
        """
        Override to load or instantiate contexts to be used to seed the self mix.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the self mix episode.

        This function will be called before the first turn of every episode.
        """
        return ['Hi!', '']

    def init_openers(self) -> None:
        """
        Override to load or instantiate opening messages to be used to seed the self
        mix.
        """
        if self.opt.get('seed_messages_from_task'):
            self._openers, self._source_list = load_openers(self.opt)
        elif self.opt.get('seed_messages_from_file'):
            self._openers = load_openers_from_file(self.opt['seed_messages_from_file'])

    def get_openers(self, episode_num: int) -> Optional[List[str]]:
        """
        Override to return one or more opening messages with which to seed the self mix
        episode.

        The return value should be an array of strings, each string being a message in
        response to the string before it.
        """

        if self._openers:
            return [open_msg for open_msg in self._openers[episode_num]]# [random.choice(self._openers)]
        return None

    def compute_metrics(self, eval_pred):
        """
        Compute metric function for skill classifier
        """
        logits, labels = eval_pred
        metric = load_metric("accuracy")
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def init_skill_classifier(self):
        """
        Initialize skill classifier model, trainer & tokenizer
        """
        # training_args = TrainingArguments("test_trainer")
        training_args = TrainingArguments(
            "test_trainer",
            per_device_eval_batch_size=1,
            per_device_train_batch_size=1
        )
        model = AutoModelForSequenceClassification.from_pretrained("/home/minju/skill_classifier/output", num_labels=3)
        self.skill_classifier = Trainer(
            model=model,
            args=training_args,
            compute_metrics=self.compute_metrics,
        )
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def reset(self):
        """
        Reset all agents in the world, and world statistics.
        """
        # for (leader, follower) in self.agents:
        #     leader.reset()
        #     follower.reset()
        for team in self.agents:
            for agent in team:
                agent.reset()
        self.max_exs = None
        self.total_exs = 0
        self.total_epochs = 0
        self.total_parleys = 0
        self.time.reset()

    # def write(self):
    #     pass

    def display(self):
        s = super().display()
        if self.turn_cnt == 0:
            s += '\n==============================\n'
        return s

    def episode_done(self):
        return self.turn_cnt >= self.max_turn_cnt

    def _get_seed_utt_acts(
        self, episode_num: int, agents: List[Agent]
    ) -> List[Dict[str, Any]]:
        openers = self.get_openers(episode_num)
        if not openers:
            return []

        actions = []
        for (leader, follower) in agents:
            action = [{'text': openers[0], 'episode_done': False, 'id': 'seed'},
                      {'text': openers[1], 'episode_done': False, 'id': 'seed'}]
            actions.append(action)
        return actions   

    def parley(self):
        debug = False

        if self.episode_done():
            self.turn_cnt = 0
            self.episode_cnt += 1
            self.contexts = None
            self.seed_utterances = None
            self.dialogue_history = []
            agents = self.get_agents()
            for i in range(self.nsubtask):
                for j in [0, 1]:
                    agents[i][j].reset()
            for i in range(len(self.retrieval_experts)):
                self.retrieval_experts[i].reset()

        if debug:
            print('\nepisode_cnt', self.episode_cnt)
            print('turn_cnt', self.turn_cnt)
            # print('agents', self.agents)

        if self.turn_cnt == 0:
            self.dialogue_history = []
            self.acts = [[None] * 2 for _ in range(self.nsubtask)]
            # get the beginning of the conversation, which can include contexts
            # and/or any number of starting messages
            self.contexts = np.array(self.get_contexts(self.episode_cnt))
            self.seed_utterances = self._get_seed_utt_acts(
                self.episode_cnt, self.agents
            )
            self.active_flags = [0, 0, 0]
            self.active_flags[self.task_to_index[self._source_list[self.episode_cnt]]] = 1
        if self.contexts is not None:
            assert len(self.contexts) == self.nsubtask

            # initial context
            for i in range(self.nsubtask):
                for j in [0, 1]:
                    context = Message(
                        {'text': self.contexts[i][j], 'episode_done': False, 'id': 'context'} # self.contexts[i][j] = skill context
                    )
                    self.acts[i][j] = context
                    self.agents[i][j].observe(validate(context))
                    if debug: print(f'context[{i}][{j}]', self.acts[i][j]) 
            # clear contexts so they are only added once per episode
            self.documents = copy.deepcopy(self.contexts)
            self.contexts = None

        elif self.seed_utterances:
            # pop the next two seed messages (there may be less or more than 2 total)
            assert len(self.seed_utterances) == self.nsubtask
            utts = self.seed_utterances
            # print('seed utt', utts)
            self.seed_utterances = None
            # print('self.seed_utt', self.seed_utterances)

            # for i, seed_utt in enumerate(self.seed_utterances):
            #     print(f'seed_utterance{i}', seed_utt)

            # process the turn
            for i in range(len(self.agents)):
                for j in [0, 1]: # leading utterance and following utterance
                    self.acts[i][j] = utts[i][j]
                    if j == 0:
                        if hasattr(self.agents[i][j], 'self_observe'):
                            self.agents[i][j].observe({'episode_done': False})
                            self.agents[i][j].self_observe(self.acts[i][j]) # this observes it's own uttereance
                        else:
                            self.acts[i][j] = self.agents[i][j].act() 
                        self.agents[i][1 - j].observe(validate(self.acts[i][j])) # observing the opponent's seed utterance
                    else:
                        self.agents[i][1 - j].observe(validate(self.acts[i][j])) # observing the opponent's seed utterance
                        if hasattr(self.agents[i][j], 'self_observe'):
                            self.agents[i][j].observe({'episode_done': False})
                            self.agents[i][j].self_observe(self.acts[i][j]) # this observes it's own uttereance
                        else:
                            self.acts[i][j] = self.agents[i][j].act() 
                    
                    if debug: print(f'seed uttererance pairs [{i}][{j}]', self.acts[i][j])

            self.dialogue_history.append(self.acts[0][0])
            self.dialogue_history.append(self.acts[0][1])

            if debug:
                print('\nContexts and seeds are initialized. Starting bot2bot conversation.\n')

        else:
            # do regular loop
            acts = self.acts
            agents = self.agents


            # Leaders action
            response_candidates = []
            for i in range(len(self.agents)):
                acts[i][0] = agents[i][0].act()
                # the number of predicted beam may not reach to the given beam size
                if len(acts[i][0]['beam_texts']) < self.opt['beam_size']:
                    beam_texts = copy.deepcopy(acts[i][0]['beam_texts'])
                    for j in range(len(acts[i][0]['beam_texts']), self.opt['beam_size'] ):
                        beam_texts.append(('', -1e4))
                    acts[i][0].force_set('beam_texts', beam_texts)
                response_candidates.append(acts[i][0]['beam_texts'])
            if debug: input('Leaders actioned\n')
            
            # Leaders debate
            verdicts = self.filter_out(response_candidates, self.documents[:,0]) # TODO provide only leader's context
            decisions, rank_scores, score_dist, (expertise_id, beam_id) = self.decide(response_candidates, verdicts, self.documents[:,0])
            for i, (verdict, decision, rank_score, dist) in enumerate(zip(verdicts, decisions, rank_scores, score_dist)):
                acts[i][0].force_set('verdict', ','.join(list(map(str, verdict))))
                acts[i][0].force_set('decision', ','.join(list(map(str, decision))))
                acts[i][0].force_set('rank_score',','.join(list(map(str, rank_score))))
                acts[i][0].force_set('score_dist', dist)
            if debug: input('Leaders debated\n')

            if debug:
                for i in range(len(self.agents)): print(f'self.acts[{i}][0]', self.acts[i][0], end='\n\n')
                input('Before decision updates\n')

            acts[expertise_id][0].force_set('text', response_candidates[expertise_id][beam_id][0])

            if debug: 
                print(f'\n ***The final decision is :{acts[expertise_id][0]["text"]}***\n')
                print('Decision Matrix', decisions)
                input('Updated leaders\' decision')
                for i in range(len(self.agents)): print(f'self.acts[{i}][0]', self.acts[i][0], end='\n\n')
                input('After decision updates\n')
                
            # Followers observe
            for i in range(len(self.agents)):
                agents[i][1].observe(validate(acts[expertise_id][0]))
                self.dialogue_history.append(acts[expertise_id][0])
            if debug: input('Followers observed\n')
            
            ### ================== turn switching =================== ###
            
            # Followers action
            response_candidates = []
            for i in range(len(self.agents)):
                acts[i][1] = agents[i][1].act()
                # the number of predicted beam may not reach to the given beam size
                if len(acts[i][1]['beam_texts']) < self.opt['beam_size']:
                    beam_texts = copy.deepcopy(acts[i][1]['beam_texts'])
                    for j in range(len(acts[i][1]['beam_texts']), self.opt['beam_size'] ):
                        beam_texts.append(('', -1e4))
                    acts[i][1].force_set('beam_texts', beam_texts)
                response_candidates.append(acts[i][1]['beam_texts'])
            if debug: input('Followers actioned\n')

            # Followers debate
            verdicts = self.filter_out(response_candidates, self.documents[:,1])
            decisions, rank_scores, score_dist, (expertise_id, beam_id) = self.decide(response_candidates, verdicts, self.documents[:,1])
            for i, (verdict, decision, rank_score, dist) in enumerate(zip(verdicts, decisions, rank_scores, score_dist)):
                acts[i][1].force_set('verdict', ','.join(list(map(str, verdict))))
                acts[i][1].force_set('decision', ','.join(list(map(str, decision))))
                acts[i][1].force_set('rank_score',','.join(list(map(str, rank_score))))
                acts[i][1].force_set('score_dist', dist)
            if debug: input('Followers debated\n')
            
            if debug: 
                for i in range(len(self.agents)): print(f'self.acts[{i}][1]', self.acts[i][1], end='\n\n')
                input('Before decision updates\n')
            
            acts[expertise_id][1].force_set('text', response_candidates[expertise_id][beam_id][0])

            if debug:
                print(f'\n ***The final decision is :{acts[expertise_id][1]["text"]}***\n')
                print('Decision Matrix', decisions)
                input('Updated followers\' decision')
                for i in range(len(self.agents)): print(f'self.acts[{i}][1]', self.acts[i][1], end='\n\n')
                input('After decision updates\n')
                
            # Leaders observe
            for i in range(len(self.agents)):
                agents[i][0].observe(validate(acts[expertise_id][1]))
                self.dialogue_history.append(acts[expertise_id][1])
            if debug: input('Leaders observed\n')

        self.update_counters()
        self.turn_cnt += 1

    def fact_check(self, premise, hypotheis) -> bool:
        # sentence-pair NLI for fact check
        with torch.no_grad():
            # label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
            tokens = ROBERTA.encode(premise, hypotheis)
            if len(tokens) > 512:
                tokens = tokens[:512]
            score = ROBERTA.predict('mnli', tokens)
            prediction = score.argmax().item()
        
        if prediction == 0:
            verdict = 0
        else:
            verdict = 1

        return verdict

    def tokenize_and_preprocess_function(self, examples):
        label_to_id = {'convai2': 0, 'empathetic_dialogues': 1, 'wizard_of_wikipedia': 2}
        result =  self.tokenizer(examples['text'], padding="max_length", truncation=True)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    def task_specific_check(self, response):
        response_set = Dataset.from_dict({'text': [response], 'label': ['convai2']})
        tokenized_response = response_set.map(self.tokenize_and_preprocess_function, batched=True)
        predictions = self.skill_classifier.predict(test_dataset=tokenized_response).predictions
        predictions = self.softmax(predictions)

        with open("/home/minju/skill_classifier/prediction_result.jsonl", 'a') as f:
            f.write(json.dumps({'text': response, 'prediction': predictions.tolist()}) + "\n")

        if(max(predictions[0])) > 0.4:
            verdict = 1
        else:
            verdict = 0
        return verdict

    def skill_classifier_func(self, response):
        response_set = Dataset.from_dict({'text': [response], 'label': ['convai2']})
        tokenized_response = response_set.map(self.tokenize_and_preprocess_function, batched=True)
        predictions = self.skill_classifier.predict(test_dataset=tokenized_response).predictions
        predictions = self.softmax(predictions)
        return predictions

    def get_entropy(self, utt1, utt2):
        skill_distribution1 = self.skill_classifier_func(utt1)[0]
        skill_distribution2 = self.skill_classifier_func(utt2)[0]
        return float(entropy(skill_distribution1, qk = skill_distribution2))

    def filter_out(self, response_candidates, contexts):
        debug = False
        short = False

        ntask = len(response_candidates)
        nbeam = len(response_candidates[0])
        virdicts = [[1] * nbeam for _ in range(ntask)]

        
        subtasks = self.opt.get('subtasks')
        
        # Cross-domain first, regarding them as more reliable filtering than cross-claim fact-checking
        if debug or short: cnt = 0
        # for i, context_pair in enumerate(contexts):
        #     for j, context in enumerate(context_pair): # TODO Decide whether we comapare a claim to both leading/following contexts
        for i, context in enumerate(contexts):
            for m, beam_texts in enumerate(response_candidates):
                for n, (claim, _score) in enumerate(beam_texts):
                    if debug or short: cnt += 1
                    if virdicts[m][n]:
                        if debug: print('FACT CHECKING CNT', cnt, f'{subtasks[i]} {subtasks[m]}-{n}')
                        if debug: print('Context', context)
                        if debug: print('Claim', claim)
                        if claim == '': 
                            virdicts[m][n] = 0
                            continue
                        # if i == m:
                        #     if debug: print()
                        #     continue
                        if debug: print('Virdict', virdicts[m][n])
                        if debug: print()
                        if self.opt['use_skill_classifier']:
                            virdicts[m][n] &= self.task_specific_check(claim)
                        virdicts[m][n] &= self.fact_check(context, claim)
                        if short: 
                            if virdicts[m][n] == 0: print(f'Cross-domain contradiction on fact-checking #{cnt}\nContext - {subtasks[i]}:\n{context}\nClaim - {subtasks[m]} #{n+1}:\n\t{claim} => Filtered Out\n')
        if debug or short: print('Virdicts after cross-domain', virdicts, '\n')


        # Getting the agreements for dialogue consistency
        if debug or short:
            agreements = [response_candidates[i][j][0] for i, task in enumerate(virdicts) for j, virdic in enumerate(task) if virdic == 1]
            print("\nExpert's Agreements")
            print('\n'.join(agreements)) if agreements else print('None')
            print()

        return virdicts

    def decide(self, response_candidates, virdicts, documents):
        num_agents = len(response_candidates)
        beam_size = len(response_candidates[0])
        
        # Create response candidate file for fixed candidate
        response_candidates_list = []
        data_path = os.path.dirname(self.opt.get('outfile'))
        f = open(self.opt.get('outfile')[:-4] + '_response_candidates.txt', 'w')
        for i in range(num_agents):
            for j in range(beam_size):
                f.write(response_candidates[i][j][0] + '\n')
                response_candidates_list.append(response_candidates[i][j][0])
        f.close()

        # skill-aware rankers observe history and sCtx
        retrieval_results = []
        for i in range(num_agents):
            if self.active_flags[i] :
                active_agent_idx = i
                context = Message({'text': documents[i], 'episode_done': False, 'id': 'context'})
                self.retrieval_experts[i].set_fixed_candidates(False)
                self.retrieval_experts[i].observe(validate(context))
                for msg in self.dialogue_history:
                    self.retrieval_experts[i].observe(validate(msg))
                retrieval_results.append(self.retrieval_experts[i].act()['text_candidates'])
            else:
                retrieval_results.append([''] * len(response_candidates_list))

        score_distribution = []
        score = virdicts
        min_score = 9999
        select_rank1 = 9999
        for i in range(num_agents):
            ranks_by_agent = []
            for j in range(beam_size):
                try:
                    ranks = []
                    for k in range(num_agents):
                        if self.active_flags[k]:
                            ranks.append(retrieval_results[k].index(response_candidates_list[i * beam_size + j]) + 1)
                            score[i][j] += score[i][j] * self.active_flags[k] * ranks[k]
                        else:
                            ranks.append(0)
                    ranks_by_agent.append(ranks)
                except:
                    score[i][j] = 0
                if score[i][j] != 0 and score[i][j] < min_score and j < select_rank1:
                    min_score = score[i][j]
                    max_row = i
                    max_col = j
                    select_rank1 = j
            score_distribution.append(ranks_by_agent)

        if min_score == 9999:
            max_row = 0
            max_col = 0
            select_rank1 = 0

        print("min_score", min_score)
        print(max_row)
        print(max_col)

        decimat = np.zeros_like(virdicts)
        decimat[max_row][select_rank1] = 1

        if max_row != active_agent_idx:
            if self.get_entropy(self.dialogue_history[-1]['text'], response_candidates[max_row][max_col][0]) <= 3:
                pass
            else:
                max_row = active_agent_idx

        
        # Oracle removes active flag and activates agent
        for i in range(num_agents):
            if i == max_row:
                self.active_flags[i] = 1
            else:
                self.active_flags[i] = 0

        return decimat, score, score_distribution, (max_row, max_col)