# LLM-use-cases

This repository showcases some Large Language model use cases, from translation, summarization and Retrieval Augmented Generation (RAG) using technologies like Hugging Face, OpenAI and AWS Bedrock

# Projects

## audio-to-text

In this project I'm interested in a few topics: transcribing an audio file and translating and summarizing it - all using Foundational models availible on Hugging Face

## hugging-face


In this project I will use a variety of pre-trained Hugging Face LLMs to complete a series of tasks, including classifying the sentiment in a car’s text review, answering a customer question, summarizing and translating text.

## multimodel-rag

This project demonstrates how to implement a multi-modal Retrieval-Augmented Generation (RAG) system using Amazon Bedrock with Amazon Nova and LangChain. Many documents contain a mixture of content types, including text and images. Traditional RAG applications often lose valuable information captured in images. With the emergence of Multimodal Large Language Models (MLLMs), we can now leverage both text and image data in our RAG systems.

## reviews-summarization

In this project I will be using a LLM to summarise hotel reviews into specific topics and also used it for automatic topic discovery

# Setup 

The requiremets.txt file contains all the python libraries that were used for these projects. The easiest way t oreprodce these results is to make use of a virtual environment (e.g conda) as follows

`conda create --name <env_name> --file requirements.txt`
