"""Simple streaming JSON (Schema) parser"""

from enum import Enum
import json
from typing import Union

from pyrsistent import PRecord, PSet, s as pset, field

class Candidates(Enum):
    NONE = 0
    ANY = 1

def reduce_candidates(*args: Union[Candidates, list[str]]):
    out = []
    for arg in args:
        if arg == Candidates.NONE:
            continue
        if arg == Candidates.ANY:
            return Candidates.ANY
        out += arg
    if len(out) == 0:
        return Candidates.NONE
    return out

class NodeParser(PRecord):
    """Base class for parsers"""
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init():
        """Create a parser"""

    @staticmethod
    def step(state: Union['NodeParser', None], char: str):
        """Returns fresh state if the provide char is valid"""

    @staticmethod
    def candidates(state: Union['NodeParser', None]) -> Union[Candidates, list[str]]:
        """Returns the set of possible candidates for next token"""

def parser_for_type(node, schema):
    """Returns a perse to handle a particular type"""
    if '$ref' in node:
        name = node['$ref'].split('/')[-1]
        return parser_for_type(schema['definitions'][name], schema)

    if 'enum' in node:
        return UnionParser.init([LiteralParser.init(json.dumps(v)) for v in node['enum']], schema)

    if 'anyOf' in node:
        return UnionParser.init([parser_for_type(v, schema) for v in node['anyOf']], schema)

    if 'type' in node:
        if node['type'] == 'string':
            return StringParser.init()
        if node['type'] == 'number':
            return NumberParser.init()
        if node['type'] == 'array':
            return ListParser.init(node, schema)
        if node['type'] == 'object':
            return DictParser.init(node, schema)


class NumberParser(NodeParser):
    so_far: str = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init():
        return NumberParser(so_far='', completed=False, can_continue=True)
    
    @staticmethod
    def step(state: Union['NumberParser', None], char: str):
        if not state:
            return None 

        is_digit = char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        is_period = char == '.'

        if len(char) == 0:
            if not is_digit:
                return None

        if is_period and '.' in state.so_far:
            return None

        if not is_digit and not is_period:
            return None

        return state.update({
            'so_far': state.so_far + char,
            'completed': True,
            'can_continue': True
        })

    @staticmethod
    def candidates(state: Union['NumberParser', None]):
        if '.' in state.so_far:
            return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

class StringState(Enum):
    """Different states in the String Parser state machine"""
    OPENING = 1
    OPENED = 2
    ESCAPING = 3
    COMPLETE = 4

class StringParser(NodeParser):
    """Parser for fully dynamic string (that can be anything)"""

    state: StringState = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init():
        """Set up an empty string parser"""
        return StringParser(state=StringState.OPENING, completed=False, can_continue=True)

    @staticmethod
    def step(state: Union['StringParser', None], char: str):
        """Step through one character"""
        if not state:
            return None

        if state.state == StringState.OPENING:
            if char in [' ', '\t', '\n', '\r']:
                return state

            if char == '"':
                return state.set('state', StringState.OPENED)

            return None

        if state.state == StringState.OPENED:
            if char == '\\':
                return state.set('state', StringState.ESCAPING)
            if char == '"':
                return state.update({'state': StringState.COMPLETE, 'completed': True, 'can_continue': False})
            return state

        if state.state == StringState.ESCAPING:
            return state.set('state', StringState.OPENED)

    @staticmethod
    def candidates(state: Union['StringParser', None]):
        if not state:
            return Candidates.NONE
        if state.state == StringState.OPENING:
            return ['"']
        if state.state == StringState.OPENED:
            return Candidates.ANY
        if state.state == StringState.ESCAPING:
            return Candidates.ANY
        if state.state == StringState.COMPLETE:
            return Candidates.NONE

class ListState(Enum):
    """Different states used in the list parser state machine"""
    OPENING = 1
    MAYBE_ELEM = 2
    DEFINE_ELEM = 3
    ELEM_NEXT = 4
    COMPLETE = 5

class ListParser(PRecord):
    """Parser for lists"""

    state: ListState = field()
    completed: bool = field()
    can_continue: bool = field()
    node = field()
    schema = field()
    active_element_state: NodeParser = field()

    @staticmethod
    def init(node, schema):
        """Set up an empty list parser"""
        return ListParser(
            state=ListState.OPENING,
            completed=False,
            can_continue=True,
            node=node,
            schema=schema)

    @staticmethod
    def step(state: Union['ListParser', None], char: str):
        """Step through one character"""
        if not state:
            return None

        # If we're not delegating characters to the element, skip whitespace
        if state.state != ListState.DEFINE_ELEM:
            if char in [' ', '\t', '\n', '\r']:
                return state
            
        if state.state == ListState.OPENING:
            if char == '[':
                return state.update({
                    'state': ListState.MAYBE_ELEM,
                    'active_element_state': parser_for_type(state.node['items'], state.schema)
                })
            return None

        if state.state == ListState.MAYBE_ELEM:
            if char == ']':
                return state.update({
                    'state': ListState.COMPLETE, 
                    'completed': True,
                    'can_continue': False,
                    'active_element_state': None
                })
            
            state = state.update({
                'state': ListState.DEFINE_ELEM
            })
        
        if state.state == ListState.DEFINE_ELEM:
            item = state.active_element_state.step(state.active_element_state, char)
            if not item:
                # If this isn't part of the child item, then check if we considered it
                # as if we were in ELEM_NEXT, if it would be a valid char and return that
                if state.active_element_state.completed:
                    state = state.update({
                        'state': ListState.ELEM_NEXT, 
                        'active_element_state': None
                    })
                    return state.step(state, char)

                return None

            state = state.set('active_element_state', item)
            if item.completed and not item.can_continue:
                state = state.update({
                    'state': ListState.ELEM_NEXT, 
                    'active_element_state': None
                })

            return state
        
        if state.state == ListState.ELEM_NEXT:
            if char == ',':
                return state.update({
                    'state': ListState.DEFINE_ELEM, 
                    'active_element_state': parser_for_type(state.node['items'], state.schema)
                })
            if char == ']':
                return state.update({
                    'state': ListState.COMPLETE,
                    'completed': True,
                    'can_continue': False
                })
            return None

    @staticmethod
    def candidates(state: Union['ListParser', None]) -> Union[Candidates, list[str]]:
        if not state:
            return Candidates.NONE
        
        if state.state == ListState.OPENING:
            return ['[']

        if state.state == ListState.MAYBE_ELEM:
            item_candidates = state.active_element_state.candidates(state.active_element_state)
            if item_candidates == Candidates.ANY:
                return Candidates.ANY
            if item_candidates == Candidates.NONE:
                return [']']
            return item_candidates + [']']

        if state.state == ListState.DEFINE_ELEM:
            element_candidates = state.active_element_state.candidates(state.active_element_state)
            if state.active_element_state.completed:
                return reduce_candidates(element_candidates, [',', ']'])
            return element_candidates
        
        if state.state == ListState.ELEM_NEXT:
            return [',', ']']

        if state.state == ListState.COMPLETE:
            return Candidates.NONE

class DictState(Enum):
    """Different states used in the dict parser state machine"""
    OPENING = 1
    OPENED = 2
    PICK_PROP = 3
    PROP_COLON = 4
    PROP_DEF = 5
    PROP_NEXT = 6
    COMPLETE = 7

class DictParser(PRecord):
    """Parser for dicts/objects"""

    state: DictState = field()
    active_prop: Union[str, None] = field()
    active_prop_state: Union[NodeParser, None] = field()
    defined_props: PSet = field()
    valid_props: PSet = field()
    completed: bool = field()
    can_continue: bool = field()
    node = field()
    schema = field()

    @staticmethod
    def init(node, schema):
        """Set up an empty list parser"""
        return DictParser(
            state=DictState.OPENING, 
            active_prop=None,
            active_prop_state=None,
            defined_props=pset(),
            valid_props=pset(*node['properties'].keys()),
            completed=False,
            can_continue=True,
            node=node,
            schema=schema)

    @staticmethod
    def step(state: Union['DictParser', None], char: str):
        """Step through one character"""
        if not state:
            return None

        # Handle whitespace
        if state.state in [DictState.OPENING, DictState.OPENED, DictState.PROP_COLON, DictState.PROP_NEXT]:
            if char in [' ', '\t', '\n', '\r']:
                return state

        if state.state == DictState.OPENING:
            if char == "{":
                return state.set('state', DictState.OPENED)
            
        if state.state == DictState.OPENED:
            if char == '"':
                return state.update({'state': DictState.PICK_PROP, 'active_prop': ''})

        if state.state == DictState.PICK_PROP:
            if char == '"':
                if state.active_prop in state.valid_props:
                    return state.update({
                        'state': DictState.PROP_COLON,
                        'active_prop_state': parser_for_type(state.node['properties'][state.active_prop], state.schema)
                    })
                return None
            next_prop = state.active_prop + char
            if any([p.startswith(next_prop) for p in state.valid_props]):
                return state.set('active_prop', next_prop)
            return None

        if state.state == DictState.PROP_COLON:
            if char == ':':
                return state.set('state', DictState.PROP_DEF)
            return None

        if state.state == DictState.PROP_DEF:
            item = state.active_prop_state.step(state.active_prop_state, char)
            if not item:
                # If this isn't part of the child item, then check if we considered it
                # as if we were in PROP_NEXT, if it would be a valid char and return that
                if state.active_prop_state.completed:
                    state = state.update({
                        'state': DictState.PROP_NEXT, 
                        'valid_props': pset(*[k for k in state.valid_props if k != state.active_prop]),
                        'defined_props': state.defined_props.add(state.active_prop),
                        'active_prop_state': None,
                        'active_prop': None
                    })
                    return state.step(state, char)

                return None
            
            if item.completed and not item.can_continue:
                return state.update({
                    'state': DictState.PROP_NEXT,
                    'valid_props': pset(*[k for k in state.valid_props if k != state.active_prop]),
                    'defined_props': state.defined_props.add(state.active_prop),
                    'active_prop_state': None,
                    'active_prop': None
                })

            return state.set('active_prop_state', item)

        if state.state == DictState.PROP_NEXT:
            if char == ',':
                if len(state.valid_props):
                    return state.set('state', DictState.OPENED)
                return None
            if char == '}':
                if all([r in state.defined_props for r in state.node['required']]):
                    return state.update({
                        'state': DictState.COMPLETE,
                        'completed': True,
                        'can_continue': False
                    })
                return None
            return None

    @staticmethod
    def candidates(state: Union['DictParser', None]) -> Union[Candidates, list[str]]:
        if not state:
            return Candidates.NONE

        if state.state == DictState.OPENING:
            return ['{ ']
        if state.state == DictState.OPENED:
            return ['"' + p + '": ' for p in state.valid_props]
        if state.state == DictState.PICK_PROP:
            return [p[len(state.active_prop):] + '":' for p in state.valid_props if p.startswith(state.active_prop)]
        if state.state == DictState.PROP_COLON:
            return [': ']
        if state.state == DictState.PROP_DEF:
            prop_candidates = state.active_prop_state.candidates(state.active_prop_state)
            if state.active_prop_state.completed:
                if len(state.valid_props) > 1:
                    return reduce_candidates(prop_candidates, [', ', ' }'])
                return reduce_candidates(prop_candidates, [' }'])
            return prop_candidates

        if state.state == DictState.PROP_NEXT:
            if len(state.valid_props):
                return [', ', ' }']
            return [' }']
        if state.state == DictState.COMPLETE:
            return Candidates.NONE

class LiteralParser(NodeParser):
    goal: str = field()
    so_far: str = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init(value):
        """Set up an empty literal parser"""
        return LiteralParser(goal=value, so_far='', completed=False, can_continue=True)

    @staticmethod
    def step(state: Union['LiteralParser', None], char: str):
        if not state:
            return None

        if state.goal == state.so_far + char:
            return state.update({
                'so_far': state.goal,
                'completed': True,
                'can_continue': False
            })

        if state.goal.startswith(state.so_far + char):
            return state.update({
                'so_far': state.so_far + char
            })

        return None

    @staticmethod
    def candidates(state: Union['LiteralParser', None]) -> Union[Candidates, list[str]]:
        if not state or state.completed:
            return Candidates.NONE

        return [state.goal[len(state.so_far):]]

class UnionParser(NodeParser):
    """Parser for Unions"""

    branches: list[NodeParser] = field()
    schema = field()
    completed: bool = field()
    can_continue: bool = field()

    @staticmethod
    def init(branches, schema):
        """Set up an empty union parser"""
        return UnionParser(branches=branches, schema=schema, completed=False, can_continue=True)

    @staticmethod
    def step(state: Union['UnionParser', None], char: str):
        if not state:
            return None

        branches = list(filter(lambda b: b is not None, map(lambda b: b.step(b, char), state.branches)))
        if len(branches) == 0:
            return None

        completed = any(b.completed for b in branches)
        can_continue = any(b.can_continue for b in branches)

        return state.update({
            'branches': branches,
            'completed': completed,
            'can_continue': can_continue
        })

    @staticmethod
    def candidates(state: Union['LiteralParser', None]) -> Union[Candidates, list[str]]:
        if not state:
            return Candidates.NONE

        return reduce_candidates(*[b.candidates(b) for b in state.branches])
    
    
from transformers import StoppingCriteria, LogitsProcessor, NoRepeatNGramLogitsProcessor, LogitsProcessorList
from pydantic import BaseModel
import json
from pprint import pprint
from typing import Literal
import numpy as np
import torch

class StreamingParserStoppingCriteria(StoppingCriteria):
  def __init__(self, parser, prompt):
    self.parser = parser
    self.prompt = prompt
    
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
    prefix = tokenizer.decode(input_ids[0])[len(self.prompt):]
    parser = self.parser
    for c in prefix:
      parser = parser.step(parser, c)
      if not parser:
        print("We're lost checking if we should stop!", prefix)
        return True
    return parser.completed

class StreamingParserLogitsProcessor(LogitsProcessor):
  def __init__(self, parser, prompt, prev_processor=None):
    self.parser = parser
    self.prompt = prompt
    self.prev_processor = prev_processor
    
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    if self.prev_processor:
        scores = self.prev_processor(input_ids, scores)
    
    # First build up the parser on the context so far
    prefix = tokenizer.decode(input_ids[0])[len(self.prompt):]
    parser = self.parser
    for c in prefix:
      parser = parser.step(parser, c)
      if parser.completed:
        return scores
      if not parser:
        print("We're lost!", prefix)
        return scores

    # Now iterate through the most likely next tokens and choose 
    sorted, indices = torch.sort(scores[0], descending=True)

    # Stash the previous state so we can reset back to it if the given token fails
    prev_state = parser

    for i in indices:
        
      # Iterate through each candidate and set the previously bad ones to be zeroes out
      next = tokenizer.decode(i)
        
      # If this is all whitespace and not just a space or the previous token is a space, skip it
      if len(prefix) and len(next.strip()) == 0 and len(next) > 0 and (next != ' ' or prefix[-1] == ' '):
        scores[0][i] = -float("inf")
        continue
        
      failed = False
      for c in next:
        n = parser.step(parser, c)
        if not n:
          # print("reject", repr(next), repr(prefix))
          parser = prev_state
          failed = True
          break
        parser = n
      if not failed:
        print(next, end='')
        break
      else:
        scores[0][i] = -float("inf")

    return scores
    
def create(tokenizer, model, cls, prompt):
    schema = json.loads(cls.schema_json())
    parser = parser_for_type(schema, schema)
    
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = tokens.input_ids.size(1)
    outputs = model.generate(
        **tokens,
        max_length=500,
        do_sample=False,
        num_return_sequences=1,
        logits_processor=[StreamingParserLogitsProcessor(parser, prompt)],#, prev_processor=NoRepeatNGramLogitsProcessor(2))],
        stopping_criteria=[StreamingParserStoppingCriteria(parser, prompt)],
        output_scores=True,
        return_dict_in_generate=True,
        renormalize_logits=True
    )
    
    total_score = 0
    for i in range(len(outputs['scores'])):
        s = outputs['scores'][i]
        t = outputs['sequences'][0][input_length + i].squeeze()
        total_score += s[0,t]
    
    out = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)[len(prompt):]
    return cls.parse_raw(out), np.exp((-total_score/len(outputs['scores'])).cpu())

from pydantic import BaseModel
import openai

def create_api(cls, orig_prompt, max_tokens, confidence=20, call=openai.Completion.create):
    schema = json.loads(cls.schema_json())

    orig_parser = parser_for_type(schema, schema)
    parser = orig_parser
    
    prompt = orig_prompt
    usage = 0

    i = 0
    while True:
        i += 1
        candidates = parser.candidates(parser)
        response = None
        if type(candidates) == list:
            if len(candidates) == 1:
                prompt += candidates[0]
                print(candidates[0], end="")
                for c in candidates[0]:
                    parser = parser.step(parser, c)
                i -= 1
                continue

            if len(candidates) > 1:
                candidate_tokens = list(map(tokenizer.encode, candidates))

                firsts = [c[0] for c in candidate_tokens]
                if len(set(firsts)) == 1:
                    candidate = tokenizer.decode(candidate_tokens[0][:1])
                    prompt += candidate
                    print(candidate, end="")
                    for c in candidate:
                        parser = parser.step(parser, c)
                    i -= 1
                    continue            

                bias = {
                    198: -100,
                    628: -100,
                    197: -100,
                    220: -100,
                    201: -100
                }
                for can in candidate_tokens:
                    bias[can[0]] = 100

                budget = max_tokens - usage - len(tokenizer.encode(prompt))
                if budget <= 0:
                    raise Exception("Out of budget")
                    
                response = call(model="text-davinci-003", 
                                                    prompt=prompt, 
                                                    temperature=1,
                                                    max_tokens=1, 
                                                    logit_bias=bias)
                usage += response['usage']['total_tokens']

        if not response:
            budget = max_tokens - usage - len(tokenizer.encode(prompt))
            if budget <= 0:
                raise Exception("Out of budget")
            budget = min(confidence, budget)
                    
            response = call(model="text-davinci-003", 
                                            prompt=prompt, 
                                            temperature=0,
                                            max_tokens=budget, 
                                            logit_bias={
                                                198: -100,
                                                628: -100,
                                                197: -100,
                                                220: -100,
                                                201: -100
                                            })
            usage += response['usage']['total_tokens']

        # Figure out how far through we can make it through the prompt
        response = prompt[len(orig_prompt):] + response['choices'][0]['text']
        test_parser = orig_parser
        valid = ""
        complete = False
        for c in response:
            n = test_parser.step(test_parser, c)
            if n:
                test_parser = n
                valid += c
                if n.completed:
                    complete = True
                    break
            else:
                break

        print((orig_prompt + valid)[len(prompt):], end="")

        if complete:
            print("\n\nComplete with usage:", usage, ", prompt + final token count", len(tokenizer.encode(orig_prompt + valid)))
            return cls.parse_raw(valid)

        prompt = orig_prompt + valid
        parser = test_parser