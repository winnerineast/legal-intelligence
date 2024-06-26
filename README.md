# Legal Intelligence
This is an AI assistant NOT for attorneys, paralegals but laypersons like you and me.

## How to run
1. A computer running MacOS (no Linux or Windows for time being).
2. An administrator account of your computer.
3. A python3 environment.
4. install ollama and refer to [intallation guide](https://github.com/ollama/ollama)
5. To run command "pip install -r requirements.txt" at root folder of this project.
6. coming soon...

## Design Principle
1. to leverage some craw platform to harvest internet. [craw_management](https://github.com/winnerineast/crawlab)
2. to leverage llm-answer-engine to have better craw. [search_craw](https://github.com/winnerineast/llm-answer-engine.git)
3. to vectorize the data by pure coding.[vector](https://github.com/winnerineast/rag-from-scratch)
4. to enhance RAG structure with spRAG. [spRAG](https://github.com/winnerineast/spRAG.git)
5. to have a Q & A UI with LLM to drill insights.[brain](https://github.com/winnerineast/quivr)
5. to have Reor to link various knowledge together. [brain2](https://github.com/winnerineast/reor.git)
6. to act back to internet based on insight derived from above.[agents](https://github.com/winnerineast/crewAI)

## Work Daily
### 2024-06-30
- added gptpdf that could process pdf files.
- added real-time voice AI agent (No.18 of AI-sampples)
- created version 3 based on MaxKB

### 2024-06-28
- added lazyGPT as workflow.
- added cube-studio as baseline model training platform.

### 2024-05-20
- completed version1.
- re-created version2.

### 2024-05-19
- split MaxKB source code into frontend and backend in version1.

### 2024-05-15
- added [MaxKB](https://github.com/winnerineast/MaxKB.git)
- recreate version1.

### 2024-05-14
- deleted version1 and focus on the incremental value-added things instead of repeating making wheels.
- tested successfully running local multiple nodes of scrawlab.
- start to write the first scraw task to capture news from a [blog](https://www.jiqizhixin.com/users/27999d5c-8072-4eb7-8f45-f4c1bcc1d0b9).

### 2024-05-11
- continue to read and understand the go project of craw_management in order to consolidate backend.
- added [markdowner](https://github.com/winnerineast/markdowner.git) which is a tool to convert website into LLM-ready markdown data.

### 2024-05-08
- successfully built both frontend (pnpm run build) and backend (go build) and db (docker compose up)
- to run backend default is master mode (crawlab server) and frontend (pnpm serve)
- next step is to create script to run worker mode and master mode at the same time and see if simplify the backend into one project.

### 2024-05-07
- [craw_management](https://github.com/winnerineast/crawlab) - backend - start to consolidate source code and simplify the configuration for easy deployment.
- created version1 folder to have a clear and clean code base.

### 2024-05-06
- added dify as workflow engine. [workflow](https://github.com/winnerineast/dify.git)

### 2024-05-05
- added llm-answer-engine to have better craw. [search_craw](https://github.com/winnerineast/llm-answer-engine.git)
- added second brain Reor to link various knowledge together. [brain2](https://github.com/winnerineast/reor.git)
- added spRAG to enhance performance of RAG. [spRAG](https://github.com/winnerineast/spRAG.git)
- ran through craw_management project folders and plan to make it slim.

### 2024-05-04
- prepared 4 codebase.
- prepared gemma as my coding pilot.
- next is to make all of 4 to run locally without any environment dependency.
- next of next is to link them together to have a data pipeline.