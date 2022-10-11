# #!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


import os
import re
import json
import pandas as pd
import numpy as np
import copy
import random
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import parlai.utils.logging as logging
from parlai.utils.io import PathManager
from typing import Tuple, Dict, List

from math import isclose
from collections import OrderedDict

# from parlai.scripts.eval_model import eval_model

from eval_model import eval_model

RESOURCES = {
    'convai2': DownloadableFile(
        'http://parl.ai/downloads/convai2/convai2_fix_723.tgz',
        'convai2_fix_723.tgz',
        'd0ae89defe2fd0b0a4221eaa642a457d7d40cef475f54798119c7f3b8dd9361d',
    ), 
    'wizard_of_wikipedia': DownloadableFile(
        'http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz',
        'wizard_of_wikipedia.tgz',
        '2a549627a83fea745efa2076a41d1c0078ad002ab2b54eae6a4e3d3d66ae24b7',
    ), 
    'empatheticdialogues': DownloadableFile(
        'http://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz',
        'empatheticdialogues.tar.gz',
        '56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d',
    )
}

SPLIT_RATIO = OrderedDict({'train': 0.8, 'valid': 0.1, 'test': 0.1})

def build(opt):
    version = 'v0.0'
    dpath = os.path.join(opt['datapath'], 'pbst')
    
    if not build_data.built(dpath, version):
        subtaskpaths = []

        for subtask in opt['subtasks']:
            subtaskpath = os.path.join(opt['datapath'], subtask)
            subtaskpaths.append(subtaskpath)
        
        logging.info('building data: ' + dpath)
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for subtask, subtaskpath in zip(opt['subtasks'], subtaskpaths):
            if not build_data.built(subtaskpath):
                build_data.make_dir(subtaskpath)
                downloadable_file = RESOURCES[subtask]
                downloadable_file.download_file(subtaskpath) 

        # Mark the data as built.
        for subtaskpath in subtaskpaths:
            build_data.mark_done(subtaskpath, version)

        if 'empatheticdialogues' in opt['subtasks']:
            # Move empatheticdialogues to parent directory
            # (ED 데이터셋만 내부폴더가 하나 더 생긴다. tar.gz라서 그런듯.)
            from shutil import move
            ed_path = subtaskpaths[opt['subtasks'].index('empatheticdialogues')]
            srcdir = os.path.join(ed_path, 'empatheticdialogues')
            if os.path.isdir(srcdir):
                for filename in os.listdir(srcdir):
                    move(os.path.join(srcdir, filename), os.path.join(ed_path, filename))
                os.rmdir(os.path.join(ed_path, 'empatheticdialogues'))

        context, random_candidates = _build_context_and_response(opt, subtaskpaths)
        
        blended_context_path = os.path.join(dpath, 'blended_context.jsonl')
        with open(blended_context_path, 'w') as fout:
            # json.dump(context, fout)
            for dic in context:
                json.dump(dic, fout) 
                fout.write("\n")

        context_splits = _split(context, dpath, SPLIT_RATIO, randomized=True)
        

        _create_parlai_format(dpath, opt)

        # Mark the data as built.
        build_data.mark_done(dpath, version)



def _convai_parser(filepath
    ) -> Tuple[List[str], List[str], List[str], List[str]]:

    debug = False

    print('Parsing ConvAI2 on', filepath)
    leading_contexts, following_contexts, seed_list, responses = [], [], [], []
    
    # Pre-processing
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        lines[i] = re.sub(r"^[0-9]+", "", line).strip()
    
    # Collecting
    persona1, persona2, seed_pair = [], [], None
    episode_done = False
    all_seed_pairs = []
    
    def roleswap(persona_list):
        for i, persona_sent in enumerate(persona_list):
            if persona_sent.startswith("partner's persona: "):
                persona_list[i] = persona_sent.replace("partner's persona: ", "your persona: ")
            else:
                persona_list[i] = persona_sent.replace("your persona: ", "partner's persona: ")
        return persona_list
    
    for i, line in enumerate(lines):
        if debug and i < 50:
            print('Line', i, line) # for debug
        if line.startswith("partner's persona: "):
            persona1.append(line)
        elif line.startswith("your persona: "):
            persona2.append(line)
            episode_done = False
        elif not episode_done:
            seed_pair = line.split('\t')
            assert len(seed_pair) == 2
            if seed_pair[0] == '__SILENCE__':
                nextline = lines[i+1]
                nextpair = nextline.split('\t')
                seed_list.append([seed_pair[1], nextpair[0]])
                leading_contexts.append('\n'.join(roleswap(persona2))) 
                following_contexts.append('\n'.join(roleswap(persona1)))
                all_seed_pairs.append([seed_pair[1], nextpair[0]])
            else:
                seed_list.append(seed_pair)
                leading_contexts.append('\n'.join(persona1)) 
                following_contexts.append('\n'.join(persona2))
                all_seed_pairs.append(seed_pair)
            responses.extend([seed_pair[0], seed_pair[1]])
            episode_done = True
            persona1, persona2, seed_pair = [], [], []
        else:
            utt_pair = line.split('\t')
            responses.extend([utt_pair[0], utt_pair[1]])
            all_seed_pairs.append([utt_pair[0], utt_pair[1]])
            

    if debug:
        for i, (leading_context, following_context, seed) in enumerate(zip(leading_contexts, following_contexts, seed_list)):
            if i == 2:
                break
            print(leading_context)
            print(following_context)
            print(seed)
            input()
        for i, response in enumerate(responses):
            if i == 20:
                break
            print(response)
            input()

    return leading_contexts, following_contexts, seed_list, responses, all_seed_pairs

def _wizard_of_wikipedia_parser(filepath
    ) -> Tuple[List[str], List[str], List[str], List[str]]:

    debug = False

    print('Parsing wizard_of_wikipedia on', filepath)
    leading_contexts, following_contexts, seed_list, responses = [], [], [], []
    all_seed_pairs = []

    topic_list, passage_list, persona_list = [], [], []
    with open(filepath, 'r') as file:
        wow = json.load(file)
    for i, episode in enumerate(wow):
        # keys: 1 'chosen_topic', 1 'chosen_topic_passage (title?)', X 'persona', 1 'wizard_eval', X 'dialog'
        topic = episode['chosen_topic']
        persona = episode['persona']
        wizard_eval = episode['wizard_eval']
        dialog = episode['dialog']
        chosen_topic_passage = episode['chosen_topic_passage']

        if debug and i < 2:
            print('topic', topic)
            print('persona', persona)
            print('wizard_eval', wizard_eval)
            print('chosen_topic_passage', chosen_topic_passage)
            print('len passage', len(chosen_topic_passage))
            print('len dialog', len(dialog))
            print('utts', [utt['text'] for utt in dialog])


        passage = ' '.join(chosen_topic_passage) 
        # passage = chosen_topic_passage[0] 
        if dialog[0]['speaker'].endswith('Apprentice'): 
            # 1_Apprentice first and 0_Wizard second
            for i in range(0, len(dialog), 2):
                if i+1 < len(dialog):
                    seed_pair = [dialog[i]['text'], dialog[i+1]['text']]
                    all_seed_pairs.append(seed_pair)
            seed_pair = [utt['text'] for utt in dialog[:2]]
        else: 
            # 0_Wizard first and 1_Apprentice second
            for i in range(1, len(dialog), 2):
                if i+1 < len(dialog):
                    seed_pair = [dialog[i]['text'], dialog[i+1]['text']]
                    all_seed_pairs.append(seed_pair)
            seed_pair = [utt['text'] for utt in dialog[1:3]]
        topic_list.append(topic)
        passage_list.append(passage)
        persona_list.append(persona)
        seed_list.append(seed_pair)
        responses.extend([utt['text'] for utt in dialog])

    assert  len(topic_list) == len(passage_list) == len(seed_list) == len(persona_list)

    for topic, passage, persona in zip(topic_list, passage_list, persona_list):
        # leading_contexts.append(f'Topic: {topic}\nPersona: {persona}') # Persona 떼어냈다.
        leading_contexts.append(f'topic: {topic}') 
        following_contexts.append(f'topic: {topic}\nknowledge: {passage}')
    
    
    assert len(leading_contexts) == len(following_contexts) == len(seed_list)

    # for debug
    if debug:
        for i, (leading_context, following_context, seed) in enumerate(zip(leading_contexts, following_contexts, seed_list)):
            if i == 5:
                break
            print(leading_context)
            print(following_context)
            print(seed)
            input()

        for i, response in enumerate(responses):
            if i == 20:
                break
            print(response)
            input()

    return leading_contexts, following_contexts, seed_list, responses, all_seed_pairs

def _empatheticdialogues_parser(filepath
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
    
    debug = False

    print('Parsing empatheticdialogues on', filepath)
    leading_contexts, following_contexts, seed_list, responses = [], [], [], []
    all_seed_pairs = []

    situation_list, emotion_list, seed_list = [], [], []

    # Preprocessing
    df = pd.read_csv(filepath, usecols=range(8), sep=',', lineterminator='\n', quotechar="`")
    df['prompt'] = df['prompt'].str.replace('_comma_', ',')
    df['utterance'] = df['utterance'].str.replace('_comma_', ',')
    
    # Collecting
    situation_list = df.groupby('conv_id').agg({'prompt':lambda x: list(x)[0]})['prompt']
    emotion_list = df.groupby('conv_id').agg({'context':lambda x: list(x)[0]})['context']
    seed_list = df.groupby('conv_id').agg({'utterance':lambda x: list(x)[:2]})['utterance']
    responses = df['utterance']

    for i in range(0, len(df['utterance']), 2):
        if i + 1 < len(df['utterance']):
            all_seed_pairs.append([df['utterance'].iloc[i], df['utterance'].iloc[i+1]])

    # Check
    remove_idx = []
    for i, seed in enumerate(seed_list):
        # Some episodes have only one turn! So we drop the such rows, 
        # as we cannot get a pair of initial utterances
        if len(seed) != 2: 
            remove_idx.append(i)
    situation_list = situation_list.drop(situation_list.index[remove_idx]).tolist()
    emotion_list = emotion_list.drop(emotion_list.index[remove_idx]).tolist()
    seed_list = seed_list.drop(seed_list.index[remove_idx]).tolist()
    responses = responses.tolist() # We don't when we're gathering utterances

    assert len(situation_list) == len(emotion_list) == len(seed_list)

    # Added extra information of emotion labels to the leader. Fair enough?
    for situation, emotion in zip(situation_list, emotion_list):
        leading_contexts.append(f'situation: {situation}\nemotion: {emotion}')
        following_contexts.append(f'')
    assert len(leading_contexts) == len(following_contexts) == len(seed_list)

    if debug:
        for i, (leading_context, following_context, seed) in enumerate(zip(leading_contexts, following_contexts, seed_list)):
            if i == 2:
                break
            print(leading_context)
            print(following_context)
            print(seed)
            input()
        for i, response in enumerate(responses):
            if i == 20:
                break
            print(response)
            input()
    return leading_contexts, following_contexts, seed_list, responses, all_seed_pairs

def parser_switch():
    parser_switch = {
        'convai2': {
            'files': ['train_both_original_no_cands.txt', 'valid_both_original_no_cands.txt'],
            'func': _convai_parser,
        },
        'wizard_of_wikipedia': {
            'files': ['train.json', 'valid_random_split.json', 'test_random_split.json'],
            'func': _wizard_of_wikipedia_parser,
        },
        'empatheticdialogues': {
            'files': ['train.csv', 'valid.csv', 'test.csv'],
            'func': _empatheticdialogues_parser,
        }
    }
    return parser_switch

def _parse_task_dataset(subtask, subtaskpath
    ) -> Tuple[List[str], List[str], List[List[str]]]:
    # Collect the context and the seed utterance pair of each episode
    leading_contextss, following_contextss, seeds, responses = [], [], [], []
    all_seed_pairs = []

    # Identify correct parser and iterate all files to parse
    parser = parser_switch()[subtask]
    for file in parser['files']:
        filepath = os.path.join(subtaskpath, file)
        fleading_contextss, ffollowing_contextss, fseeds, fresponses, fall_seed_pairs = parser['func'](filepath)
        leading_contextss.extend(fleading_contextss)
        following_contextss.extend(ffollowing_contextss)
        seeds.extend(fseeds)
        responses.extend(fresponses)
        all_seed_pairs.extend(fall_seed_pairs)
    return leading_contextss, following_contextss, seeds, responses, all_seed_pairs

def _retrieve_contextual_document(origin_opt, seed_queries, contextual_docs, teacher, origin, target, subtaskpath):
    '''
        retreival: query (seed utterances) -> document (contexts)
    '''
    retrieved_doc = []

    # Semantic Retreival (e.g. poly-encoder, DPR)
    if teacher == 'semantic':
        parlai_data_path = subtaskpath[:subtaskpath.find('pbst')]

        opt = {}
        if target == 'convai2':
            opt['task'] = 'persona_inference:retrieval'
        elif target == 'wizard_of_wikipedia':
            opt['task'] = 'topic_inference:retrieval'
        elif target == 'empatheticdialogues':
            opt['task'] = 'emotion_inference:retrieval'
        else:
            raise RuntimeError('Unimplemented subtask')

        split = opt['task'].split(':')

        opt['model'] = 'transformer/biencoder'
        opt['eval_candidates'] = 'inline'
        opt['fixed_candidates_path'] = parlai_data_path + split[0] + '/fixed_candidates.txt'
        opt['batchsize'] = 256
        opt['datatype'] = 'retrieval'
        opt['world_logs'] = parlai_data_path + split[0] + '/retrieval_report.json'
        opt['report_filename'] = parlai_data_path + split[0] + '/retrieval_report.json'
        opt['log_keep_fields'] = 'all'
        opt['num_examples'] = -1
        opt['display_examples'] = False
        opt['save_format'] = 'conversations'

        eval_list = []

        candidates_path = parlai_data_path + split[0] + '/fixed_candidates.txt'
        f = open(candidates_path, 'r')
        candidates = f.readlines()
        f.close()

        for query in seed_queries:
            # input_dict = {'text': query, 'label_candidates': candidates}
            input_dict = {'text': query, 'labels': candidates[0]}
            eval_list.append(input_dict)
            
        with open(parlai_data_path + split[0] + '/retrieval.json', "w") as json_file:
            json.dump(eval_list, json_file)
        print("Saved queries to", parlai_data_path + split[0] + '/retrieval.json')

        eval_model(opt)

        # Open retrieval result (jsonl file)
        retrieval_result_path = opt['report_filename'] + 'l'

        with open(retrieval_result_path, 'r') as json_file:
            json_list = list(json_file)
        retrieval_result = []
        for json_str in json_list:
            result = json.loads(json_str)
            retrieval_result.append(result)

        retrieved_doc = []
        for retrieved in retrieval_result:
            retrieved_doc.append(retrieved['dialog'][0][1]['text'])
            

    # Random Retrieval
    elif teacher == 'random':
        doc_ids = list(range(len(contextual_docs)))
        retrieved_doc_idx = random.choices(doc_ids, k=len(seed_queries))
        retrieved_doc = contextual_docs[retrieved_doc_idx]

    # Lexical Retrieval??
    elif teacher == 'lexical_retrieval':

        parlai_data_path = origin_opt['datapath']
        src2trg = f'{origin}->{target}'

        # Loading TF-IDF Retriever Model
        tr_opt = {}
        if target.startswith('convai2'):
            task = 'persona_inference'
        elif target.startswith('wizard_of_wikipedia'):
            task = 'topic_inference'
        elif target.startswith('empatheticdialogues'):
            task = 'emotion_inference'
        else:
            raise RuntimeError('Unimplemented retrieval task')
        tr_opt['task'] = f'{task}:{teacher}'
        tr_opt['model'] = 'tfidf_retriever'
        tr_opt['model_file'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/model'
        tr_opt['eval_candidates'] = 'inline'
        tr_opt['fixed_candidates_path'] = None
        tr_opt['batchsize'] = 256
        tr_opt['datatype'] = f'{teacher}/{src2trg}/query'
        tr_opt['label_candidates_file'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/fixed_candidates.txt'
        tr_opt['world_logs'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/{src2trg}/results.jsonl'
        tr_opt['report_filename'] = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/{src2trg}/model_report.json'
        tr_opt['log_keep_fields'] = 'all'
        tr_opt['num_examples'] = -1
        tr_opt['display_examples'] = False
        tr_opt['save_format'] = 'conversations'
        tr_opt['tensorboard_log'] = False
        
        # prepare candidate for the retrieval, which are contextual documents
        with open(tr_opt['label_candidates_file'], 'r') as f:
            candidates = f.readlines()
        
        # prepare queries of the retireval, which are the seed utterances
        eval_list = []
        if target.startswith('wizard_of_wikipedia'):
            leading_queries = seed_queries[0]
            following_queries = seed_queries[1]
            for i in range(len(leading_queries)):
                input_dict = {'text': leading_queries[i] + ' ' + following_queries[i], 'labels': candidates[0]}
                eval_list.append(input_dict)
        else:
            for query in seed_queries:
                input_dict = {'text': query, 'labels': candidates[0]}
                eval_list.append(input_dict)
        
        # save the queries as files
        retrieval_query_path = f'{parlai_data_path}/pbst/contextual_alignment/{task}/{teacher}/{src2trg}/query.json'
        retrieval_dirpath = retrieval_query_path.rsplit('/', 1)[0]
        if os.path.exists(retrieval_dirpath):
            build_data.remove_dir(retrieval_dirpath)
        build_data.make_dir(retrieval_dirpath)
        with open(retrieval_query_path, "w+") as json_file:
            json.dump(eval_list, json_file)
        print("Saved queries to", retrieval_query_path)
 
        # run parali retreival task, which then saves the retrieval results in as a 'world_logs' file
        if os.path.exists(tr_opt['world_logs']) == False:
            print("Evaluating...")
            eval_model(tr_opt)

        # load retrieval result (as in jsonl file) to read and return the retrieved documents
        with open(tr_opt['world_logs'], 'r') as json_file:
            json_list = list(json_file)

        retrieval_result = []
        for json_str in json_list:
            result = json.loads(json_str)
            if 'text' not in result['dialog'][0][1]:
                result['dialog'][0][1]['text'] = ""
            if 'text_candidates' not in result['dialog'][0][1]:
                result['dialog'][0][1]['text_candidates'] = ['']*5
            retrieval_result.append(result)

        for retrieved in retrieval_result:
            # retrieved_doc.append(retrieved['dialog'][0][1]['text'])
            retrieved_doc.append(retrieved['dialog'][0][1]['text_candidates'])

        print('*'*5, "Contextual Alignment Example", '*'*5)
        print("Query:", seed_queries[0])
        print("Retreived Document:", retrieved_doc[0][0])
        print("Num of Retreived Document:", len(retrieved_doc[0]))
        print()

    return retrieved_doc

        


def _build_context_and_response(opt, subtaskpaths):
    # contexts are different: for leading speaker and following speaker 
    subtasks, nsubtasks = opt['subtasks'], len(opt['subtasks'])
    leading_context_dic, following_context_dic, seed_dic = {}, {}, {} # seed pairs are concatenated into a sentence
    response_candidates = {}

    # Collect task-wise contexts and seeds
    for subtask, subtaskpath in zip(subtasks, subtaskpaths):
        leading_contexts, following_contexts, seeds, responses, all_seed_pairs = _parse_task_dataset(subtask, subtaskpath)
        lc = os.path.join(opt['datapath'], 'pbst', f'leading_{subtask}_kb.json')
        fc = os.path.join(opt['datapath'], 'pbst', f'following_{subtask}_kb.json')
        su = os.path.join(opt['datapath'], 'pbst', f'seed_utterance_pairs_{subtask}.json')
        rc = os.path.join(opt['datapath'], 'pbst', f'responses_candidates_{subtask}.json')
        asp = os.path.join(opt['datapath'], 'pbst', f'all_seed_utterance_pairs_{subtask}.json')
        with open(lc, 'w') as f0, open(fc, 'w') as f1, open(su, 'w') as f2, open(rc, 'w') as f3, open(asp, 'w') as f4:
            json.dump(leading_contexts, f0)
            json.dump(following_contexts, f1)
            json.dump(seeds, f2)
            json.dump(responses, f3)
            json.dump(all_seed_pairs, f4)
        assert len(leading_contexts) == len(following_contexts) == len(seeds)
        leading_context_dic[subtask] = leading_contexts
        following_context_dic[subtask] = following_contexts
        seed_dic[subtask] = all_seed_pairs
        response_candidates[subtask] = responses
        print(f'{len(seeds)} contexts were parsed from the {subtask}')
        print(f'{len(all_seed_pairs)} seed utterances pairs were parsed from the {subtask}')
        print(f'Also, {len(responses)} response candidates were parsed from the {subtask}\n')

    # writing the maximum pool of response candidate sampled from each task datasets
    response_candidates = np.array([response.strip() for response_set in response_candidates.values() for response in response_set])
    response_candidates = np.unique(response_candidates).tolist()
    rc = os.path.join(opt['datapath'], 'pbst', f'responses_candidates.json')
    with open(rc, 'w') as f:
        json.dump(response_candidates, f)
    print(f'\nTotal of {len(response_candidates)} response candidates were parsed from the given {", ".join(subtasks)} subtasks\n')

    # Contextual alignment
    lcm = leading_contextual_matrix = [[None]*nsubtasks for _ in range(nsubtasks)]
    fcm = following_contextual_matrix = copy.deepcopy(lcm)

    # Align inter-task relationhip between seeds and contexts
    for i, origin in enumerate(subtasks):
        seed_pairs = np.array(seed_dic[origin])
        leading_seeds, following_seeds = seed_pairs[:,0], seed_pairs[:,1]
        for j, (target, subtaskpath) in enumerate(zip(subtasks, subtaskpaths)):
            leading_contexts = leading_context_dic[target]
            following_contexts = following_context_dic[target]
            # Retrieve contextual document from different task
            # And align the seed with all the other subtask's context
            if target == 'convai2':
                lcm[i][j] = _retrieve_contextual_document(opt, leading_seeds, leading_contexts, 'lexical_retrieval', origin, target+'_1', subtaskpath)
                fcm[i][j] = _retrieve_contextual_document(opt, following_seeds, following_contexts, 'lexical_retrieval', origin, target+'_2', subtaskpath)
            elif target == 'wizard_of_wikipedia':
                lcm[i][j] = _retrieve_contextual_document(opt, [leading_seeds, following_seeds], leading_contexts, 'lexical_retrieval', origin, target+'_1', subtaskpath)
                fcm[i][j] = copy.deepcopy(lcm[i][j])
                for k in range(len(lcm[i][j])):
                    lcm[i][j][k] = lcm[i][j][k].split('\n')[0]
            else:
                lcm[i][j] = _retrieve_contextual_document(opt, leading_seeds, leading_contexts, 'lexical_retrieval', origin, target+'_1', subtaskpath)
                fcm[i][j] = [''] * len(lcm[i][j])

    context = []

    for srctaskid, srctask in enumerate(subtasks):

        context_length = len(lcm[srctaskid][0])
        alignedseed = seed_dic[srctask]
        
        for context_id in range(context_length):
            episode = {}
            episode['source_task'] = srctask
            episode['leader'] = {}
            episode['follower'] = {}

            episode['leader']['seed'] = alignedseed[context_id][0]
            episode['follower']['seed'] = alignedseed[context_id][1]
            episode['leader']['context'] = {}
            episode['follower']['context'] = {}

            for alignedtaskid, alignedtask in enumerate(subtasks):
                try:
                    episode['leader']['context'][alignedtask] = lcm[srctaskid][alignedtaskid][context_id] # 여기서 오류남
                except:
                    raise RuntimeError

            for alignedtaskid, alignedtask in enumerate(subtasks):
                episode['follower']['context'][alignedtask] = fcm[srctaskid][alignedtaskid][context_id]

            context.append(episode)
    
 
    return context, response_candidates

def _split(json_list, dpath, split_ratio: OrderedDict, randomized=True):
    data = json_list
    ds = dataset_size = len(json_list)

    assert isclose(sum([v for v in split_ratio.values()]), 1) # escape overflow
    ss = splitset_size = {k: round(ds * v) for k, v in split_ratio.items()}

    # Random sampling
    if randomized:
        random.shuffle(data)

    def greedy_split(data, sample_ratio):
        split_index = int(round(len(data) * sample_ratio, 6)) # 6의 자리 반올림
        sampled_data = data[:split_index]
        remained_data = data[split_index:]
        return sampled_data, remained_data

    original_length = len(data)
    sd = split_data = OrderedDict({split_name: None for split_name in split_ratio.keys()})

    remained_ratio = 1
    for i, (split_name, split_ratio) in enumerate(split_ratio.items()):
        sample_ratio = split_ratio / remained_ratio
        sd[split_name], data = greedy_split(data, sample_ratio)
        remained_ratio -= split_ratio

    assert isclose(remained_ratio, 0, abs_tol=1e-5), "Errors in split ratio"
    split_lengths = [len(d) for d in sd.values()]
    assert sum(split_lengths) == original_length, "Some samples in datset is remained after split"

    for split_name, dataset in sd.items():
        bc = os.path.join(dpath, f'blended_context_{split_name}.jsonl')
        pbc = os.path.join(dpath, f'pretty_blended_context_{split_name}.jsonl')
        with open(bc, 'w') as outputfile, open(pbc, 'w') as prettyfile:
            for sample in dataset:
                assert isinstance(sample, Dict)
                json.dump(sample, outputfile)
                outputfile.write('\n')
                json.dump(sample, prettyfile, indent=4)
                prettyfile.write('\n')
    return split_data


def _create_parlai_format(dpath: str, opt: List):
    """
    Copy data into the format read by ParlAIDialogTeacher.

    'text' will be from the free Turker, who speaks first, and 'label' will be from the
    guided Turker.
    """
    datatypes = ['train', 'valid', 'test']
    for datatype in datatypes:

        load_path = os.path.join(dpath, f'blended_context_{datatype}.jsonl')
        save_path = os.path.join(dpath, f'blended_context_{datatype}.txt')

        print(f'Loading {load_path}.')
        data = []
        with PathManager.open(load_path, 'r', encoding='utf8') as f_read:
            for line in f_read:
                data.append(json.loads(line))

        print(f'Saving to {save_path}')
        subtasks = opt['subtasks']
        with PathManager.open(save_path, 'w', encoding='utf8') as f_write:
            for episode in data:
                num_entries = 1
                entry_idx = 0
                for entry_idx in range(num_entries):
                    line = _get_line(
                        episode, num_entries, entry_idx, subtasks
                    )
                    f_write.write(f'{line} \n')


def _get_line(episode: dict, num_entries: int, entry_idx: int, subtasks: List) -> str:
    """
    Return the line to print in the reformatted file.
    """
    episode_done = entry_idx == num_entries - 1

    if entry_idx == 0:
        leader_context = '\n'.join([f"{episode['leader']['context'][task]}" for task in subtasks])
        follower_context = '\n'.join([f"{episode['follower']['context'][task]}" for task in subtasks])
        context_dataset = f"context dataset: {episode['source_task']}"
        original_context = '\n'.join([leader_context, follower_context, context_dataset]) + '\n'

    else:
        original_context = ''
    input_utterance = episode['leader']['seed']
    model_label = episode['follower']['seed']
    source_task = episode['source_task']

    # Compile into text string
    parts = {
        'text': input_utterance,
        'labels': model_label,
        'source_task': source_task
    }
    assert all([isinstance(part, str) for part in parts.values()])
    line = '\t'.join([f'{key}:{_escape(value)}' for key, value in parts.items()])

    # Add episode_done
    if episode_done:
        line += '\tepisode_done:True'

    return line


def _escape(value: str) -> str:
    return value.replace('\t', '\\t').replace('\n', '\\n').replace('|', '__PIPE__')
