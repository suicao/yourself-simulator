# DIY chatbot from your Facebook data and pretrained language models
![](https://scontent.fsgn5-15.fna.fbcdn.net/v/t39.30808-6/277738725_5261827560549331_8321740197298223750_n.jpg?_nc_cat=111&ccb=1-5&_nc_sid=730e14&_nc_ohc=S9O1ec36Mq0AX9wEKHt&tn=7hyRwwmFcJX8j_Be&_nc_ht=scontent.fsgn5-15.fna&oh=00_AT-QhqdSjoI897rofBFrddZxXf2_-7mHyTxY1X771IacXQ&oe=62628CEE)
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
Go to https://www.facebook.com/dyi/?referrer=yfi_settings to download an archive of your past data. Select the json format and low media quality for a smaller archive as we don't need the media files anyway.

Uncheck everything but the "Messages" box, Request your download and wait a few days for your archive to be available.

Unzip your data and run the following command:

```commandline
python preprocess.py --input_path /<path-to-your-data>/inbox --output_path ./data/convs.json
```
The output format should look like `./data/sample.json`

### Training
Run the following command:
```commandline
python  train.py --output_dir=output --model_type=gpt2 --do_train --model_name_or_path "NlpHUST/gpt-neo-vi-small" --block_size 128 --per_device_train_batch_size=16 --per_device_eval_batch_size=36 --gradient_accumulation_steps=4 
--save_total_limit=5 --learning_rate=2e-5 --num_train_epochs=5 --save_steps=500  --overwrite_output_dir  --train_data_file=./data/convs.json --logging_steps 500 --output_dir output2 --seed 42069
```

There are a few candidates for the pretrained Vietnamese model, here I picked `NlpHUST/gpt-neo-vi-small`, you may consider:

- https://huggingface.co/danghuy1999/gpt2-viwiki
- https://huggingface.co/imthanhlv/gpt2news
- https://huggingface.co/VietAI/gpt-neo-1.3B-vietnamese-news or https://huggingface.co/VietAI/gpt-j-6B-vietnamese-news (very large models)

### Inference
Run the following command to start a convesation with your trained model
```commandline
python infer.py  --model_name_or_path "NlpHUST/gpt-neo-vi-small" --checkpoint_path ./output/pytorch_model.bin
```
