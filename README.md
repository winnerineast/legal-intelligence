# Legal Intelligence
This is a AI assistant NOT for attorneys, paralegals but laypersons like you and me.

## How to run
1. A computer running Windows or MacOS.
2. An administrator account of your computer.
3. A python3 environment.
4. install ollama and refer to [intallation guide](https://github.com/ollama/ollama)
5. To run command "pip install -r requirements.txt" at root folder of this project.
6. to run command "streamlit run agent/app.py"
7. When the server is up and running, access the app at: http://localhost:8501
8. enjoy!

## what if...
1. what if you want to change your AI from ollama to other online APIs? answer: to integrate with [OneAPI](https://github.com/songquanpeng/one-api) and I'll do it later.
2. what if you have your own law data? answer: to put your data in folder "law_data" of this project (support pdf files for time being) and no sub-folder is accepted for time being and run command "python ./agent/process_raw_data.py" and wait for the process is done before you start to ask the chatbot.
