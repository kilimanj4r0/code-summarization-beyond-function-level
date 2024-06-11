# Modified ClassEval

# Method-level + few-shot
python3 llms-inference.py ../data mce method all dsc-1.3b
python3 llms-inference.py ../data mce method all dsc-6.7b
python3 llms-inference.py ../data mce method all dsc-33b
python3 llms-inference.py ../data mce method all sc-15b
python3 llms-inference.py ../data mce method all ll-8b

# Class-level
python3 llms-inference.py ../data mce class 0 dsc-1.3b
python3 llms-inference.py ../data mce class 0 dsc-6.7b
python3 llms-inference.py ../data mce class 0 dsc-33b
python3 llms-inference.py ../data mce class 0 sc-15b
python3 llms-inference.py ../data mce class 0 ll-8b


# Modified CodeSearchNet

# Method-level + few-shot
python3 llms-inference.py ../data mcsn method all dsc-1.3b
python3 llms-inference.py ../data mcsn method all dsc-6.7b
python3 llms-inference.py ../data mcsn method all dsc-33b
python3 llms-inference.py ../data mcsn method all sc-15b
python3 llms-inference.py ../data mcsn method all ll-8b

# Repo-level + few-shot
python3 llms-inference.py ../data mcsn repo 0,2,10 dsc-1.3b
python3 llms-inference.py ../data mcsn repo 0,2,10 dsc-6.7b
python3 llms-inference.py ../data mcsn repo 0,2,10 dsc-33b  # requires device_map='auto'
python3 llms-inference.py ../data mcsn repo 0,2,10 sc-15b  # requires device_map='auto'
python3 llms-inference.py ../data mcsn repo 0,2,10 ll-8b