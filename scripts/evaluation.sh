# Modified ClassEval

# LMs (method-level)
python3 evaluation.py ../data/predicted/code_trans_t5_large_source_code_summarization_python_multitask_finetune mce
python3 evaluation.py ../data/predicted/code_trans_t5_large_code_documentation_generation_python_multitask_finetune mce
python3 evaluation.py ../data/predicted/codet5-base-multi-sum mce
python3 evaluation.py ../data/predicted/codet5p_220m_py_sum mce
python3 evaluation.py ../data/predicted/pile-t5-large-codexglue mce

# LLMs (method-level + few-shot)
python3 evaluation.py ../data/predicted/deepseek-coder-1.3b-instruct mce --few-shot True
python3 evaluation.py ../data/predicted/deepseek-coder-6.7b-instruct mce --few-shot True
python3 evaluation.py ../data/predicted/deepseek-coder-33b-instruct mce --few-shot True
python3 evaluation.py ../data/predicted/starcoder2-15b-instruct-v0.1 mce --few-shot True
python3 evaluation.py ../data/predicted/Llama-3-8B-Instruct-Gradient-1048k mce --few-shot True

# LLMs (class-level)
python3 evaluation.py ../data/predicted/deepseek-coder-1.3b-instruct mce --level class
python3 evaluation.py ../data/predicted/deepseek-coder-6.7b-instruct mce --level class
python3 evaluation.py ../data/predicted/deepseek-coder-33b-instruct mce --level class
python3 evaluation.py ../data/predicted/starcoder2-15b-instruct-v0.1 mce --level class
python3 evaluation.py ../data/predicted/Llama-3-8B-Instruct-Gradient-1048k mce --level class


# Modified CodeSearchNet

# LMs (method-level)
python3 evaluation.py ../data/predicted/code_trans_t5_large_source_code_summarization_python_multitask_finetune mcsn
python3 evaluation.py ../data/predicted/code_trans_t5_large_code_documentation_generation_python_multitask_finetune mcsn
python3 evaluation.py ../data/predicted/codet5-base-multi-sum mcsn
python3 evaluation.py ../data/predicted/codet5p_220m_py_sum mcsn
python3 evaluation.py ../data/predicted/pile-t5-large-codexglue mcsn

# LLMs (method-level + few-shot)
python3 evaluation.py ../data/predicted/deepseek-coder-1.3b-instruct mcsn --few-shot True
python3 evaluation.py ../data/predicted/deepseek-coder-6.7b-instruct mcsn --few-shot True
python3 evaluation.py ../data/predicted/deepseek-coder-33b-instruct mcsn --few-shot True
python3 evaluation.py ../data/predicted/starcoder2-15b-instruct-v0.1 mcsn --few-shot True
python3 evaluation.py ../data/predicted/Llama-3-8B-Instruct-Gradient-1048k mcsn --few-shot True

# LLMs (repo-level + few-shot)
python3 evaluation.py ../data/predicted/deepseek-coder-1.3b-instruct mcsn --level repo --few-shot True
python3 evaluation.py ../data/predicted/deepseek-coder-6.7b-instruct mcsn --level repo --few-shot True
python3 evaluation.py ../data/predicted/deepseek-coder-33b-instruct mcsn --level repo --few-shot True
python3 evaluation.py ../data/predicted/starcoder2-15b-instruct-v0.1 mcsn --level repo --few-shot True
python3 evaluation.py ../data/predicted/Llama-3-8B-Instruct-Gradient-1048k mcsn --level repo --few-shot True