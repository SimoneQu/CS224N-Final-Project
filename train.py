import argparse
import json
import os
from collections import OrderedDict
import torch
import copy
import csv
import util
import numpy as np
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args, get_debug_args

from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(tokenizer, dataset_dict, dir_name, dataset_name, split,recompute_features):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples



#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.args = args
        self.args.train_datasets = self.args.train_datasets.split(',') #convert this to list
        self.args.eval_datasets = self.args.eval_datasets.split(',') #convert this to list

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                # TODO: compute loss

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, tokenizer):
        args = self.args
        if args.do_finetune or (args.do_train and args.run_name == "baseline"):
            if args.do_finetune:
                assert len(args.train_datasets) == 1, "finetune should be one dataset at a time"
            print("Preparing training data...")
            train_dataset, _ = get_dataset(
                args.train_datasets, args.train_dir, tokenizer, 'train', args.recompute_features)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=RandomSampler(train_dataset)
            )
            print("Preparing validation data...")
            val_dataset, val_dict = get_dataset(
                args.train_datasets, args.val_dir, tokenizer, 'val', args.recompute_features)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=SequentialSampler(val_dataset)
            )
            print("Start training...")
            best_scores = self._train_baseline(model, train_loader, val_loader, val_dict)
        elif args.run_name == "maml":
            print("Preparing training data...")
            train_loaders = dict()
            example_count = dict()
            for dataset in args.train_datasets:
                train_dataset, _ = get_dataset([dataset], args.train_dir, tokenizer, 'train', args.recompute_features)
                example_count[dataset] = len(train_dataset)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    sampler=RandomSampler(train_dataset)
                )
                train_loaders[dataset] = train_loader
            print("Preparing validation data...")
            val_dataset, val_dict = get_dataset(args.train_datasets, args.val_dir, tokenizer, 'val', args.recompute_features)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                sampler=SequentialSampler(val_dataset)
            )
            self.total_train_examples = sum(example_count.values())
            task_weights = self.get_task_weights(example_count)
            print("Start training...")
            best_scores = self._train_maml(model, train_loaders, val_loader, val_dict, task_weights)
        else:
            raise Exception

        return best_scores

    def _train_maml(self, model, train_dataloaders, eval_dataloader, val_dict, task_weights):

        ###########################
        # Pretrain using Meta-Learning
        ###########################
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.args.meta_lr)
        submodel = copy.deepcopy(model)
        submodel.to(device)
        optim_sub = AdamW(submodel.parameters(), lr=self.args.inner_lr)

        global_idx = 0
        last_eval_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        #  we are not currently going through the each entire dataset,
        #  instead we are sampling from it
        num_outer_updates = int(self.num_epochs * self.total_train_examples /
                                self.args.num_inner_updates / self.args.num_tasks / self.args.batch_size)
        with torch.enable_grad(), tqdm(total=num_outer_updates) as progress_bar:
            for global_idx in range(num_outer_updates):
                # self.log.info(f'Epoch: {epoch_num}')
                tasks = self.sample_tasks(task_weights)
                inner_info = self.init_inner_info(model, device)
                for task in tasks:
                    # Reseting the theta back to the meta theta
                    submodel.load_state_dict(model.state_dict())

                    # Inner loop
                    for i in range(self.args.num_inner_updates):
                        example = train_dataloaders[task].__iter__().next()
                        optim_sub.zero_grad()
                        submodel.train()
                        # submodel.zero_grad()
                        input_ids = example['input_ids'].to(device)
                        attention_mask = example['attention_mask'].to(device)
                        start_positions = example['start_positions'].to(device)
                        end_positions = example['end_positions'].to(device)
                        outputs = submodel(
                            input_ids,
                            attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions
                        )
                        loss = outputs[0]
                        # if (global_idx % 1) == 0:
                        # print('inner_loop', task, i, loss.data)
                        self.log.info(f'outer_loop: {global_idx}, inner_loop: {task}, {i}, {loss.data}')

                        loss.backward()
                        optim_sub.step()

                    self.update_inner_info(submodel, inner_info, train_dataloaders[task], device)

                self.meta_update(model, optim, inner_info)

                # Get best score, similar to _train_baseline
                progress_bar.update(1)
                progress_bar.set_postfix(num_outer_update=global_idx, NLL=loss.item())
                tbx.add_scalar('train/NLL', loss.item(), global_idx)
                if (global_idx % self.eval_every) == 0 and global_idx > 1:
                    self.log.info(f'Evaluating after {global_idx} rounds of MAML...')
                    preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                    self.log.info('Visualizing in TensorBoard...')
                    for k, v in curr_score.items():
                        tbx.add_scalar(f'val/{k}', v, global_idx)
                    self.log.info(f'Eval {results_str}')
                    if self.visualize_predictions:
                        util.visualize(tbx,
                                       pred_dict=preds,
                                       gold_dict=val_dict,
                                       step=global_idx,
                                       split='val',
                                       num_visuals=self.num_visuals)
                    if curr_score['F1'] >= best_scores['F1']:
                        best_scores = curr_score
                        self.save(model)
                global_idx += 1

        # Storing away the final parameters for later use
        self.save(model)

        return best_scores

    def _train_baseline(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0 and global_idx > 1:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores

    def init_inner_info(self, model, device):
        if self.args.meta_update in ["reptile", "fomaml"]:
            info = dict()
            for key in model.state_dict():
                info[key] = torch.zeros_like(model.state_dict()[key]).to(device)
        return info

    def update_inner_info(self, model, info, dl, device):
        if self.args.meta_update == "reptile":
            # info is a dict storing the sum of theta
            for key in model.state_dict():
                info[key] += model.state_dict()[key]
        elif self.args.meta_update == "original":
            example = dl.__iter__().next()

            model.zero_grad()
            model.train()

            input_ids = example['input_ids'].to(device)
            attention_mask = example['attention_mask'].to(device)
            start_positions = example['start_positions'].to(device)
            end_positions = example['end_positions'].to(device)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs[0]
            loss.backward()

            for key, param in model.named_parameters():
                info[key] += param.grad


    def meta_update(self, model, optim, inner_info):
        if self.args.meta_update == "reptile":
            for key in model.state_dict():
                model.state_dict()[key] += self.args.meta_lr * (
                        inner_info[key] / self.args.num_tasks - model.state_dict()[key])
        elif self.args.meta_update == "fomaml":
            # for key in model.state_dict():
            #     model.state_dict()[key] -= self.args.meta_lr * inner_info[key] / self.args.num_tasks
            optim.zero_grad()
            for k, v in model.named_parameters():
                v.grad = inner_info[k] / self.args.num_tasks
            optim.step()

    def get_task_weights(self, example_count):
        task_weights = dict()
        if self.args.task_weight == "proportional":
            total_examples = sum(example_count.values())
            for key in example_count:
                task_weights[key] = example_count[key] / total_examples
        elif self.args.task_weight == "unif":
            total_tasks = len(self.args.train_datasets)
            for key in self.args.train_datasets:
                task_weights[key] = 1 / total_tasks
        else:
            raise Exception

        return task_weights


    def sample_tasks(self, task_weights={}):
        tasks = np.random.choice(
            list(task_weights.keys()), self.args.num_tasks, p=list(task_weights.values()))
        return tasks

def get_dataset(datasets, data_dir, tokenizer, split_name, recompute_features):
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}')
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    data_encodings = read_and_process(tokenizer, dataset_dict, data_dir, dataset_name, split_name, recompute_features)
    return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

def main():
    # define parser and arguments
    args = get_train_test_args()
    # args = get_debug_args("maml", "original")

    util.set_seed(args.seed)
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train or args.do_finetune:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        proc = "train" if args.do_train else "finetune"
        args.save_dir = util.get_save_dir(args.save_dir, f"{args.run_name}_{proc}")
        log = util.get_logger(args.save_dir, f'log_{proc}')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        best_scores = trainer.train(model, tokenizer)
        print(f"Finished {proc}, the best score is {best_scores}")
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(
            args.eval_datasets, args.eval_dir, tokenizer, split_name, args.recompute_features)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
