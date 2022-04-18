# DIY chatbot from your Facebook data and pretrained language models

## How this works
This model generates replies based on DialoGPT style language modeling, by concatenating all dialog turns in a conversation into a long text.

For example, the follow conversation:

> Person: Do you want the Aladeen news or the Aladeen news?
> 
> You: The Aladeen news?
> 
> Person: You're HIV-Aladeen.
> 
> You: 😮

Will be transformed to the following format:

`<s> Do you want the Aladeen news or the Aladeen news? </s>  The Aladeen news? <s> You're HIV-Aladeen.<s/> 😮`

We introduce two special tokens `<s>` and `</s>`, where `<s>` denotes the beginning of a reply by another person, and `</s>` by you.

Given that the training input is just a text sequence, it can be modeled using any causal language model and used to generate a reply based on the current context.
Formally, we concatenate all dialog turns within a dialogue session into a long text 
<img src="https://render.githubusercontent.com/render/math?math=x_1, x_2, ..x_N ">,, We denote the source sentence (dialogue history)
as <img src="https://render.githubusercontent.com/render/math?math=S = x_1, x_2, ..x_M "> where <img src="https://render.githubusercontent.com/render/math?math=x_M "> is the `</s>` token, and target sentence (ground truth response) as <img src="https://render.githubusercontent.com/render/math?math=T = x_{m%2b1}, x_{m%2b2}, ..x_N ">, the conditional probability of <img src="https://render.githubusercontent.com/render/math?math=P(T|S)"> can be written as the
product of a series of conditional probabilities:

<img src="https://render.githubusercontent.com/render/math?math=P(T|S) = \prod_{n=m%2b1}^{N} p(x_n|x_1,...x_{n-1})">

## Training 
### Prepare your data
Go to https://www.facebook.com/dyi/?referrer=yfi_settings to download an archive of your past data. Select the json format and low media quality for a smaller file as we don't need the media files anyway.

Uncheck everything but the Messages box, Request your download and wait a few days.