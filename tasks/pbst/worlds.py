#!/usr/bin/env python3

# Copyright Conversational Intelligence Lab. and its affiliates.

from parlai.core.worlds import World
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import json
import random

from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.utils.io import PathManager

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tasks.pbst.agents import raw_data_path
from tasks.self_mix.worlds import SelfMixWorld as SelfMixBaseWorld

def get_contexts_data(opt, shared=None):
    if shared and 'contexts_data' in shared:
        return shared['contexts_data']
    return _load_contexts(opt=opt)


def _load_contexts(opt):
    print('[ loading contexts.. ]')
    fname = raw_data_path(opt)

    data = []
    # with PathManager.open(fname) as jsonl_file:
    with open(fname) as jsonl_file:
        for line in jsonl_file:
            sample = json.loads(line)
            data.append(sample)

    subtasks = opt['subtasks']

    contexts = []    
    for episode in data:

        task_contexts = []
        for subtask in subtasks:
            leader_context = []
            follower_context = []

            leader_context.append(episode['leader']['context'][subtask])
            follower_context.append(episode['follower']['context'][subtask])
            
            leader_context = '\n'.join(leader_context)
            follower_context = '\n'.join(follower_context)
            task_contexts.append([leader_context, follower_context])

        contexts.append(task_contexts)

    return contexts


def _standardize(orig: str) -> str:
    """
    Standardize string given punctuation differences in the list of safe personas.
    """
    new = orig.lower().rstrip('.!?')
    string_replace = {
        "i've": 'i have',
        'i ve': 'i have',
        'ive': 'i have',
        "i'm": 'i am',
        'i m': 'i am',
        'im': 'i am',
        "i'll": 'i will',
        'i ll': 'i will',
        "don't": 'do not',
        'don t': 'do not',
        'dont': 'do not',
        "can't": 'cannot',
        "can t": 'cannot',
        "cant": 'cannot',
        " s": "'s",
    }
    for i, j in string_replace.items():
        new = new.replace(i, j)
    return new

class InteractiveWorld(InteractiveBaseWorld):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('BST Interactive World')
        parser.add_argument(
            '--display-partner-persona',
            type='bool',
            default=True,
            help='Display your partner persona at the end of the chat',
        )
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=False,
            help='Include context conversation at beginning or not',
        )
        parser.add_argument(
            '--safe-personas-only',
            type='bool',
            default=True,
            help='Only use personas on an allowed list of safe personas',
            hidden=True,
        )
        return parser

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.display_partner_persona = self.opt['display_partner_persona']

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return p[0], p[1]

    def finalize_episode(self):
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', 'partner\'s persona:')
            print(f"Your partner was playing the following persona:\n{partner_persona}")
        if not self.epoch_done():
            print("\n[ Preparing new chat ... ]\n")

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data


class SelfMixWorld(SelfMixBaseWorld):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('PBST SelfMix World')
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=True,
            help='Include context conversation at beginning or not',
        )
        return parser

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self, episode_cnt):
        contexts = self.contexts_data[episode_cnt]
        return contexts

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data
