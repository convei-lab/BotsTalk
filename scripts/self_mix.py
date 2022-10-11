#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Allows a model to self-chat on a given task.
"""
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.worlds import create_task
# from parlai.tasks.self_mix.world_logging import DebateLogger
from parlai.utils.misc import TimeLogger
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from parlai.core.loader import load_task_module, load_world_module

import math
import json
import random
import os

import sys
from world_logging import DebateLogger
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tasks.pbst.worlds import SelfMixWorld
import importlib


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Generate self-mix of models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-d', '--display-examples', type='bool', default=True)
    parser.add_argument(
        '--display-add-fields',
        type=str,
        default='',
        help='Display these fields when verbose is off (e.g., "--display-add-fields label_candidates,beam_texts")',
    )
    parser.add_argument(
        '-emf',
        '--expert-model-files',
        default=None,
        help='Define a different expert model for self mix',
    )
    parser.add_argument(
        '--expert-model-opt-files',
        default=None,
        help='Comma-seperated paths to file containing opts to override for each expert',
    )
    parser.add_argument(
        '-st',
        '--selfmix-task',
        type='bool',
        default=True,
        help='Create a self mix version of the task',
    )
    parser.add_argument(
        '--num-self-mixs', type=int, default=1, help='Number of self mixs to run'
    )
    parser.add_argument(
        '--selfmix-max-turns',
        type=int,
        default=6,
        help='The number of dialogue turns before self chat ends',
    )
    parser.add_argument(
        '--seed-messages-from-task',
        type='bool',
        default=False,
        help='Automatically seed conversation with messages from task dataset.',
    )
    parser.add_argument(
        '--seed-messages-from-file',
        default=None,
        help='If specified, loads newline-separated strings from the file as conversation starters.',
    )
    parser.add_argument(
        '--subtasks',
        type=str,
        help='Comma-sperated list of subtasks for team debate'
    )
    parser.add_argument(
        '--outfile', type=str, default=None, help='File to save self mix logs'
    )
    parser.add_argument(
        '--save-format',
        type=str,
        default='conversations',
        choices=['conversations', 'parlai'],
        help='Format to save logs in. conversations is a jsonl format, parlai is a text format.',
    )
    parser.add_argument(
        '--seed-range',
        type=str,
        default=None,
        help='Seed range if you want to split seed set and train in parallel',
    )
    parser.add_argument(
        '--use-skill-classifier',
        type='bool',
        default=True,
        help='Use skill classifier while filtering',
    )
    parser.add_argument(
        '--ranker-model-files',
        type=str,
        default='zoo:pretrained_transformers/model_poly/model,empathetic_dialogues_poly/model.checkpoint,wizard_of_wikipedia_poly/model.checkpoint',
        help='Put paths of ranker model files',
    )
    parser.add_argument(
        '--activate-max-turn',
        type=int,
        default=-1,
        help='If activate max turn is -1, you don\'t use activate setting, else put max turn that you want',
    )

    parser.set_defaults(interactive_mode=True)
    DebateLogger.add_cmdline_args(parser, partial_opt=None)
    return parser


def _run_self_mix_episode(opt, world, world_logger):
    bsz = opt.get('batchsize', 1)
    num_turns = opt['selfmix_max_turns']
    assert bsz == 1, "Batch size cannot be different than 1 for self-mix"
    num_parleys = math.ceil(num_turns / bsz)
    for _ in range(num_parleys):
        world.parley()
        world_logger.log(world)

    if opt['display_examples']:
        print('-- end of episode --')

    world.reset()
    world_logger.reset_world()  # flush this episode


def self_mix(opt):
    random.seed(opt['seed'])
    subtasks = opt['subtasks'] = opt['subtasks'].split(',')
    experts = opt['expert_model_files'].split(',')
    expert_opt_files = opt.get('expert_model_opt_files').split(',')

    # Create agents
    expert_agents = []
    for subtask, expert, expert_opt_file in zip(subtasks, experts, expert_opt_files):
        opt['model_file'] = expert_opt_files
        model_pair = []
        if expert is None:
            # Self mix with same model, where leader and follower models interleave in turn for annotation
            leader = create_agent(opt, requireModelExists=True)
            follower = leader.clone()
            model_pair.append(leader)
            model_pair.append(follower)
            leader.opt.log(f"{subtask} Expert Leader Opt")
            follower.opt.log(f"{subtask} Expert Leader Opt")
        else:
            # Loading several models for mixing chatbots
            for role in ['Leader', 'Follower']:
                print('\n'*2)
                print(f'***** Loading {subtask} Expert {role} *****\n')
                if expert_opt_files:
                    print(f"WARNING: Loading override opts from: {expert_opt_files}")
                    with PathManager.open(expert_opt_file) as f:
                        expert_opt = json.load(f)
                else:
                    expert_opt = {}
                expert_opt['interactive_mode'] = opt.get('interactive_mode', True)
                expert_opt['beam_size'] = opt.get('beam_size')
                print(
                    f"WARNING: Setting expert interactive mode to: {expert_opt['interactive_mode']}"
                    f"WARNING: Setting expert beam size to: {expert_opt['beam_size']}"
                )
                model = create_agent_from_model_file(expert, expert_opt)
                model.opt.log(f"{subtask} Expert {role} Opt")
                model_pair.append(model)
            
        expert_agents.append(model_pair)

    # Create skill-aware ranker agents
    expert_model_files = opt['ranker_model_files'].split(',')
    expert_models = ['transformer/polyencoder', 'transformer/polyencoder', 'transformer/polyencoder'] 
    retrieval_experts = []
    for i in range(len(subtasks)):
        ranker_opt = {}
        ranker_opt['model_file'] = expert_model_files[i]
        ranker_opt['model'] = expert_models[i]
        ranker_opt['interactive_mode'] = True
        ranker_opt['candidates'] = 'fixed'
        ranker_opt['eval_candidates'] = 'fixed'
        ranker_opt['fixed_candidates_path'] = opt['outfile'][:-4] + '_response_candidates.txt'
        ranker_opt['ignore_bad_candidates'] = True
        ranker_opt['encode_candidate_vecs'] = True
        ranker_opt['allow_missing_init_ranker_opts'] = True
        ranker_opt['fixed_candidate_vecs'] = 'replace'
        # ranker_opt['gpu'] = -1

        # Make dummy candidates for initialization
        if not os.path.exists(os.path.dirname(ranker_opt['fixed_candidates_path'])):
            os.makedirs(os.path.dirname(ranker_opt['fixed_candidates_path']))
        f = open(ranker_opt['fixed_candidates_path'], 'w')
        f.write('Hi\nHello\nNice to meet you.\n')
        f.close()

        model = create_agent_from_model_file(expert_model_files[i], ranker_opt)
        retrieval_experts.append(model)


    # Set IDs
    for i, agent_pair in enumerate(expert_agents):
        assert len(agent_pair) == 2
        for j, agent in enumerate(agent_pair):
            agent.id = agent.id + f"_{i+1}_{j+1}"

    model_id = '_'.join([agent[0].id.rsplit('_', 1)[0] for agent in expert_agents])


    my_module = importlib.import_module("tasks.pbst.worlds")
    world_class = getattr(my_module, "SelfMixWorld")

    world = world_class(opt=opt, agents=(expert_agents, retrieval_experts))
    
    # Set up world logging
    logger = DebateLogger(opt)
    log_time = TimeLogger()

    # Run some team debates.
    for i in range(opt['num_self_mixs']):
        _run_self_mix_episode(opt, world, logger)
        report = world.report()
        text, report = log_time.log(i + 1, opt['num_self_mixs'], report)
        logging.info(text)

        if i % 50 == 0:
            # Save debates
            if opt['outfile'] is None:
                dt = opt['datatype'].split(':')[0]
                outfile = os.path.join(opt['datapath'], 'pbst', f'machine_generated_{dt}.txt')
            else:
                outfile = opt['outfile']

            if opt['save_format'] == 'conversations' and hasattr(world, 'write'):
                # use self chat specific world to write conversation
                # this might be useful for logging extra contextual
                # information (like personas)
                world.write(logger, outfile)
            else:
                # use default logger write function
                logger.write(outfile, world, opt)

        
    # Save debates
    if opt['outfile'] is None:
        dt = opt['datatype'].split(':')[0]
        outfile = os.path.join(opt['datapath'], 'pbst', f'machine_generated_{dt}.txt')
    else:
        outfile = opt['outfile']

    if opt['save_format'] == 'conversations' and hasattr(world, 'write'):
        # use self chat specific world to write conversation
        # this might be useful for logging extra contextual
        # information (like personas)
        world.write(logger, outfile)
    else:
        # use default logger write function
        logger.write(outfile, world, opt)

    return logger.get_logs()


@register_script('self_mix')
class SelfMix(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return self_mix(self.opt)


if __name__ == '__main__':
    SelfMix.main()
