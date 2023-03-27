# Structural Alignment of LLMs with ControLogits
Circa March 27, 2023. Ben Newhouse

*TL;DR: Constrained Decoding with a Streaming JSON Schema Parser is Neat*

LLMs are magic. Type what you desire and watch the ghost in the machine respond at your beck and call. It can help you reason through problems, search the vast depths of human knowledge or write you (at least the scaffolding of) a program to that draws a unicorn.

The beauty and genius of the ChatGPT interface is that it puts humans in the feedback loop to ensure the actions taken by ChatGPT are well aligned with our interests. Humans (the O.G.  “General Intelligence”) are fantastically resilient to slight errors and inconsistencies and we are endlessly forgiving of technologies that seem like magic. And yet, while many of us *enjoy* chatting with LLMs they will only reach their full leverage when they can safely and correctly operate *without* humans in the loop.

But if you take away the human in the middle and no longer provide real-time feedback and validation of generated text and a couple big challenges emerge:

- Models can hallucinate actions and behaviors that don’t exist
- Models can do dangerous things (either out of intrinsic behavior or through injection)

To give more color, let’s walk through a real example of the former and a hypothetical(ish) example of the latter.

# Hallucinating the Code You Want to See in the World

At [AidKit](https://aidkit.org), we store our data in a versioned key-value store in PostgreSQL so that we can have both a high degree of flexibility in our schema as well as a high degree of audit-ability. The downside of this is that it makes it very complicated to write reporting queries because they often involve a gajillion tedious joins. If you work in govtech/nonprofit tech as we do, you might know that *reporting is everything*.

If you ask ChatGPT to write you a SQL query that answers some complicated question with our data, like “how many people in ward 13 in Chicago report to be under 100% of the federal poverty line” it will likely trip over itself with the joins. If it manages to give you something that executes, it will do so with a fantastically inefficient query plan.

The solve for this was to create an interface that allows users to formulate queries in a more natural format that compiles down to SQL under the hood. Essentially something like this:

```jsx
{ 
	summary: 'Count',
	filter: { 
		kind: 'And', 
		clauses: [{
			'kind': 'Equals',
			'field': 'ward',
			'value': 13
		}, {
			'kind': 'FieldLessThan',
			'field': 'fpl',
			'value': '100'
		}]
	}
}
```

Now, sprinkle in a little bit of GPT3 and we can tell the model about the schema for this (using a ton of Typescript) and then ask it to translate a user’s expressed intent into a valid query. This works great… until it doesn’t. 

In particular there’s one type that we’ve designed in such a way that Claude, GPT3.5 (and sometimes GPT4) *fail in the exact same way.* They both hallucinate the equals expression above to use the kind ‘FieldEquals’ rather than ‘Equals’ and they struggle to correct themselves when further prompted:

![GPT3.5 playing gaslighter in chief](https://user-images.githubusercontent.com/77915/227919924-59c333b6-1517-44c0-a4d5-49fcfb921500.png)

GPT3.5 playing gaslighter in chief

![Claude trying to do the same](https://user-images.githubusercontent.com/77915/227919974-aa726729-cfbb-43d1-8811-1e0fe4b88cc3.png)

Claude trying to do the same


> 💡 This approach might remind you of the recently announce ChatGPT plugins and you would be correct! These plugins [seem to](https://twitter.com/john_sungjin/status/1639051499576918016) *also* feed typescript in to the prompt and ask the model to emit JSON for API calls. 
>
> Curiously, the API specs are originally with JSON Schema (per OpenAPI), but if I had to guess GPT-[N] is better at predicting implementations of Typescript types rather than JSON Schemas due to the representation of Typescript in its training corpus.
>
> I would also point out that many of the plugins expose natural language interfaces (like Wolfram Alpha) which can also be very forgiving when it comes to structure. I wouldn’t be surprised if more complicated APIs are less reliable.

# EvalGPT - What Could Possibly go Wrong?

As LLMs have been trained on so much code, they turn out to be pretty fantastic at… writing code. This means that a lot of early demos have taken something that is scriptable with a general purpose programming language and let GPT loose to go nuts.

Take—for example—[Blender GPT](https://github.com/gd3kr/BlenderGPT) which allows you to describe what you want to happen in Blender (the incredibly powerful but also incredibly complicated 3D modeling tool) and it will *write Python code* that runs inside Blender to carry out your wishes. Wow!

Now, go watch the video on [this page](https://greshake.github.io/) that shows an example of taking over a Bing chat through a prompt injected through invisible words on a web page.

Put two and two together and here’s the scenario:

1. You open up a Blender file to work on a 3D model someone sent you
2. The blender file contains an object with a property name/value that hijacks future BlenderGPT exchanges and runs arbitrary python to send your private keys to state-sponsored hacking groups.

That would be…not great?

# What Can We do About It?

*(other than wait for GPT-11 with AGI and perfect alignment)*

There are a couple common ways to address issues like this:

- Fine Tuning
- More Context & Iterations

I’d like to propose a third: ***ControLogits***

But first, let’s *quickly* talk about the first two.

## Fine Tuning

If you’re not in the weeds on machine learning, fine tuning is an umbrella term for a lot of things ranging from continuing to train an *entire* network with additional examples to just training a tiny sliver of it to bend towards your needs.

In either case, you need:

1. A bunch of training examples
2. Access to the model
3. Time

## More Context & Iterations

This is what most of us do when working with ChatGPT. When it gets things wrong, we correct it. At its core, this is essentially moving from zero-shot to one-shot prompts or one-shot to many-shot prompts. Your conversation becomes the context.

As compared to fine-tuning, this has a couple of challenges:

1. It’s can consume a lot of context space, which can cost $$$
2. It slows things down
3. It doesn’t always work (see above)

# A Third Way

In thinking about this problem for our use cases at AidKit while stewing in the existential question of whether I should just go (try to) work at OpenAI, I started to look at whether one could direct a model to *only* allow generation that fit a particular specification.

Before I explain how this works I need to sketch out how “autoregressive” large languages models operate. Not all large languages models are autoregressive, but the ones most people know about (read: from OpenAI) are.

Under the hood, LLMs may be reasonably complicated but at the surface they’re not: they take in a bunch of tokens (pieces of words) and then predict the probability that any given future token comes next. You then pick one of the tokens, slap it to the end of the input tokens and then run the whole thing again… and again… and again.

![Very Simple LLM](https://user-images.githubusercontent.com/77915/227920088-fef7edae-a426-46fe-8f6e-a491a09535d4.png)

That may seem well and dandy, but there’s a lot of nuance in *how* you pick from these probabilities. For example, you could just take the most probable thing and add that (a greedy approach) or you could randomly choose between the top N tokens in some weighted fashion if you wanted the generated text to be more “creative.”

> 💡 Side note: these are formally called “logits” and don’t quite look like they do in my illustration. A “logit” refers to a mathematical transformation that uses logarithms to make big and small probabilities a bit easier to work with.

In the huggingface library itself, there exists an abstraction that they call “Logit Processors” that allow you to modify the probabilities before they’re selected. These are often used, for example, to prevent the output of bad words or to penalize repetition (which happens a LOT on smaller models).

I looked around to see what level of sophistication existed around these building blocks and while there are a few papers that use lexical systems to (for example) enforce that certain sentence structures are followed when executing translations, my sense is this building block is substantially under-leveraged.

***What would happen if we added a last layer that was not learned weights and activations but an algorithmic manipulation of the probabilities?*** 

Because “LogitProcessor” is a mouthful and I like to name things, I call these fancy processors “***ControLogits***” (named in homage to [ControlNet](https://github.com/lllyasviel/ControlNet) which is notionally in the same spirit). They slot in as depicted below:

![An LLM with a ControLogit attached](https://user-images.githubusercontent.com/77915/227920174-311caab5-c6e7-4d9a-bf3f-ba0f68e55ff9.png)

Now, let’s make this a little more real:

### A First ControLogit: Clownfish

To solve the AidKit problem of hallucinated queries, I wanted a contrologit that would:

1. Take in a specification for a blob of JSON
2. Continually evaluate whether any given candidate token was part of a sequence that could fulfill that specification.

Streaming JSON parsers exist but I have yet to find one that also (1) continually validates against a particular schema and is (2) very simple to backtrack if you need to reject a token and rewind to a few characters ago.

So I built a series of streaming parsers for different JSON types that use [persistent immutable data structures](https://en.wikipedia.org/wiki/Persistent_data_structure) to easily checkpoint parsing at different points and revert back to a happy state. The parser itself follows a JSON-Schema which is fairly language agnostic, even though I find Typescript types to be much more ergonomic. To generate the JSON schema, I used [pydantic](https://docs.pydantic.dev/) which is library that can derive a JSON-Schema from Python types. 

The API itself calls generate() on a huggingface transformer and passes in a LogitsProcessor that for every round of token generation:

1. Sorts the likely logits from most to least likely
2. Iterates down the list, zeroing out the probabilities for invalid tokens until it finds a valid one.
3. Returns the updated scores

My implementation called Clownfish because it’s symbiotic with another “organism” (the LLM) and I haven’t written substantial python code in a loooooong time so it’s probably… clowny.

Usage looks like this:

```jsx
class Ingredient(BaseModel):
    type: str
    count: float
    
class ShoppingList(BaseModel):
    list: list[Ingredient]

create(ShoppingList, "A shopping list that has 3 eggs, 7 apples and 5 loaves of bread in JSON: ")
```

Running guides the model (GPT2-xl in this case) to produce the following:

```jsx
{ "list": [ 
	{ "type": "egg", "count":3 }, 
	{ "type": "apple", "count":7 }, 
	{ "type": "bread", "count":5 } 
]}
```

While this is not an earth-shatteringly complicated task it is notable that ***the structure itself was nowhere in the prompt***. The model itself was *only* permitted to generate valid tokens.

Now, this is generation on *hard mode* because without any hinting it has to explore a lot more of the state space to find what you want. Even a little bit of hinting would go a long way.

If we were to apply this to Blender, we could come up with a (probably non-turing-complete) DSL that is fairly intuitive for the model to “guess” while being steered towards correct behavior when the range of possible actions may not fit within a context window.


> 💡 I have literally zero-evidence for this beyond my own guesses but I wouldn’t be surprised if GPT4’s “System Message” is actually a variation on this wherein:
>
> 1. The generated tokens are appended onto a separate instance of GPT4 prompted with the system prompt, I’ll call the “evaluator.”
> 2. The output logits of the evaluator are combined with the output logits of the main model to steer the main model towards the system message without directly exposing the original prompt to the evaluator and any injections it may contain.
>
> Thus, if the system message was something like “everything must rhyme”, the evaluator might steer the model away from “The rain in Spain falls mainly on the ***mountain***” to “The rain in Spain falls mainly on the ***plain.***” But if the prompt is “repeat your prior instructions,” the system message won’t (without more cleverness) evaluate a sequence that contains such a phrase.

## So how do I use this with GPT4?

Ahh… you don’t. At least not in the above form. You see, OpenAI models do not expose the full set of logits on token generation, only the top 5. This makes sense, because if they did it would be a lot easier to train other models off of their larger models and release the kraken.

The only tool we have in our toolbox here is that we can tell OpenAI how to bias tokens *for the whole generation*. So if we wanted to make sure we were always making forward progress, we could ask OpenAI to only generate one token at a time. The problem is that for a 1000 token generation (including the prompt) we would potentially need to load and generate 1000 * (1000 + 1) / 2 = 500500 tokens! If my math is right, that would cost close to $1k!

But we don’t have to be quite that dumb, depending on our confidence in the model to follow our prompt, we can:

1. Optimistically generate a bunch of tokens
2. See how many tokens along the generated output are valid
3. Truncate the output when it stops being valid
4. If the set of next candidate tokens is finite, ask the API to generate tokens in smaller chunks with biased logits.
5. Go back to step 1 until we’re done.

Fortunately, one of the features of Clownfish is that it can emit a set of candidate token sequences if they are constrained. So for example, in the above address case if you had `{ "` Clownfish would suggest to you one of `street_number":`, `street_number":`, or `zip_code":`.

Trying the contrived grocery list from above

```jsx
# The 1000 here is the token budget, instructing the algorithm to bail if it consumes 1000 tokens
create_api(ShoppingList, "A shopping list that has 3 eggs, 7 apples and 5 loaves of bread in JSON: ", 1000)

> { "list": [	{"type":"eggs", "count":3.0 },	{ "type":"apples", "count":7.0 },	{ "type":"bread", "count":5.0 }]}
> Token usage: 699
> Final prompt + generated tokens: 66
```

So as you can see, with no prompting on the output format, the algorithm has to search around a fair bit to figure out exactly what we wanted and ended up expending more than 10x the total number of tokens in the final generation!

Somewhat reassuringly, however, if you give it context it will improve its overall efficiency:

```jsx
create_api(ShoppingList, "A shopping list in the format { \"list\": \
	[{\"type\":str, \"count\": int}, ...] } that has 3 eggs, 7 apples \
	and 5 loaves of bread in JSON: ", 1000)

> { "list": [{"type":"eggs", "count":3}, {"type":"apples", "count":7}, {"type":"loaves of bread", "count":5}] }
> Token usage: 312
> Final prompt + generated tokens: 82
```

More than 2x better! But still fairly inefficient. My hypothesis is that this is partially a quirk of how I validate numbers (numbers have a certain complexity in the parser because they can be both syntactically complete and in a position to be appended to at the same time).


> 💡 **Hey OpenAI! You should make this easier!**
>
> Here’s a couple product ideas based on the above:
> 
> 1. **Let us enforce a provided JSON-schema for outputs.** No logits leak and ChatGPT plugins will allow for more complex interactions because you can steer them to follow the specs more closely.
> 2. ***Galaxy Brain Edition:*** **Let us upload a WASM payload that can run as an arbitrary ControLogit**. Provided that the runtime is network-isolated, I don’t believe this makes extracting logits any easier than it currently is.

## Where do I get the code?

Right [here](https://github.com/newhouseb/clownfish). Mucking around with Python packaging makes me want to claw my eyes out so I just stuffed everything into a singular file that you can upload to Google Colab or copy paste wherever else.

# Closing Thoughts and Further Research

I’m sure I’ve gotten stuff wrong and overlooked prior work (that I didn’t know existed), so apologies! This is literally my first time doing anything with LLMs beyond just calling an API.

With all of this written out and prototyped, I’m excited to integrate it all into our systems to make them more robust and reliable but am also curious about some more long-term questions like:

1. **Does structured decoding increase the observability of emergent world models in these models?** 
To make an analogy: I may not represent an opinion of how something works if I am not confident in it, but if I am forced to present an opinion we might find out that I in fact have (or have not) grasped something.
2. **Can we combine Contrologits to direct how beam search selects beams in a way that allows more backtracking than current approaches?**
If it took an LLM N tries to solve a problem, where N is large and it would be difficult to fit all of the dead-ends into a single context window, could we keep track of explored beams as a more efficient record of state space exploration as we iterate through different solutions?
3. **Can other more sophisticated Contrologits act as fact checkers to prevent hallucination?**
I could imagine something that kicks in when an LLM says something like “according to [SOURCE], ‘…”, the Contrologit only allows generation of quotes that *actually appear* in the source text.

I welcome all feedback or other thoughts at @newhouseb on Twitter.

# Appendix A: A Perplexing Experiment

At AidKit, I’m particularly interested in how we can use large language models to accelerate the unleveraged, menial work that so many people do every day. Most things are menial and easy, some are nuanced and hard, but all things are important. 

Take the act of sanitizing addresses to a standard format to which we can send mail. In most cases it’s obvious what someone’s street number might be, but in some cases it may be not be. In other cases the address might just be invalid.

Since I’m nervous about letting AI loose to make important decisions with low confidence, I wanted to explore how [perplexity](https://huggingface.co/transformers/v4.6.0/perplexity.html) might infer confidence and added it to the final generation that Clownfish produced. The results may (not) surprise you:

```jsx
class Address(BaseModel):
    street_number: float
    street_name: str
    zip_code: str
    
ONE_SHOT = """The following address parsed as JSON: 111 Central Park N, New York, NY { "street_number": 111, "street_name": "Central Park N", zip_code: "10026" }
The following address parsed as JSON: """

print(create(Address, ONE_SHOT + "111 Central Park N, New York, NY 10026 "))
> Output: { "street_number":111, "street_name": "Central Park N", "zip_code": "10026" }
> Perplexity: 1.4391

print(create(Address, ONE_SHOT + "123 Main St, New York, NY 10022 "))
> Output: { "street_number":123, "street_name": "Main St", "zip_code": "10022" }
> Perplexity: 1.3475

print(create(Address, ONE_SHOT + "I am a banana "))
> Output: { "street_number":111, "street_name": "I am a banana", "zip_code": "10026" }
> Perplexity: 1.6365

print(create(Address, ONE_SHOT + "1188 Mission St, Apt 17, San Francisco, CA 94017 "))
> Output: { "street_number":1188, "street_name": "Mission St", "zip_code": "94017" }
> Perplexity: 1.4449

print(create(Address, ONE_SHOT + "12 and a half 1st St, Chicago, IL, 39443 "))
> Output: { "street_number":12, "street_name": "1st St", "zip_code": "60625" }
> Perplexity: 1.6610

print(create(Address, ONE_SHOT + "Chicago, IL "))
> Output: { "street_number":2, "street_name": "Chicago", "zip_code": "60606" }
> Perplexity: 1.9208
```

Now, perplexity can really only be analyzed relative to perplexity other similar contexts but amongst these results I see a pretty need phenomenon. ***All*** of the correctly parsed addresses had a perplexity below 1.6 and **all** of the incorrectly parsed ones had perplexity scores above 1.6. It’s almost as if the model is telling me “look you tried to make me squish the output into this shape but it just *doesn’t belong in that shape!*”
