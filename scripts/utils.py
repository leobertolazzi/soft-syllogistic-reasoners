import os


def capfirst(string):
    return string[:1].upper() + string[1:]


def lowerfirst(string):
    return string[:1].lower() + string[1:]


def get_result_dictionary(dataset, model_name, test_data, training_data, icl_examples, run_id):
    results = {
        **{metric : [] for metric in ["accuracy", *dataset["test"]["type"], "correct"]}
    }
    results = {
            "meta-data" : {
                "model_name" : model_name, 
                "test_data" : test_data,
                "training_data" : training_data,
                "icl_examples" : icl_examples,
                "run_id" : run_id
            },
            "results" : results
        }
    return results


def save_results(results):
    setting = results["meta-data"]["training_data"] + "_" + str(results["meta-data"]["icl_examples"])
    # Save results path
    to_csv_metrics = []
    metrics = [key for key in results["results"]]
    for metric in metrics:
        if type(results["results"][metric]) == list: 
            to_csv_metrics.append(str(results["results"][metric][0]))
        else:
            to_csv_metrics.append(str(results["results"][metric]))
    # create path to save results
    os.makedirs("results", exist_ok=True)
    csv_path = results["meta-data"]["test_data"] + ".csv"
    csv_path = os.path.join("results", csv_path)
    # create header if file not exists
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as o:
            o.write(f"run_id,model,setting,{','.join(metrics)}\n")
    # add results
    with open(csv_path, "a") as o:
        m_name = results["meta-data"]["model_name"]
        run_id = results["meta-data"]["run_id"]
        o.write(f"{run_id},{m_name},{setting},{','.join(to_csv_metrics)}\n")


def log_training(save_name, ep_count, accuracy):
    os.makedirs("data/train_logs", exist_ok=True)
    csv_path = f"data/train_logs/{save_name}.csv"
    if ep_count == 0:
        with open(csv_path, "w") as f:
            f.write("ep,accuracy\n")
            f.write(f"{ep_count},{accuracy}\n")
    else:
        with open(csv_path, "a") as f:
            f.write(f"{ep_count},{accuracy}\n")
            

def prompt_template_chat(context, problem, icl=False):
    if icl:
        context = f"\n\nUse the examples in the following context to better answer the test examples:\n{context}"
    
    template = [
        {
            "role" : "system",
            "content" : "You will be presented with premises together with eight possible conclusions and the option 'Nothing follows'. "\
                        "Write the conclusion that logically follows given the premises or 'Nothing follows' "\
                        "if none of the other conclusions logically follow. "\
                        "Read the passage of information thoroughly and select the correct answer from the available options. "\
                        "Read the premises thoroughly to ensure you know what the premise entails."\
                        f"{context}"
        },
        {
            "role" : "user",
            "content" : f"Complete the following test example:\n\n{problem}"
        }
    ]
    
    return template


def prompt_template(context, problem, icl=False):
    if icl:
        context = f"\n\nUse the examples in the following context to better answer the test examples:\n{context}"
    
    template = "You will be presented with premises together with eight possible conclusions and the option 'Nothing follows'. "\
                "Write the conclusion that logically follows given the premises or 'Nothing follows' "\
                "if none of the other conclusions logically follow. "\
                "Read the passage of information thoroughly and select the correct answer from the available options. "\
                "Read the premises thoroughly to ensure you know what the premise entails."\
                f"{context}"\
                f"Complete the following test example:\n\n{problem}"

    return template


def clean_model_output(text):
    text = "Answer:".join(text.split("Answer:")[:-1]) + "Answer:" + ".".join([i for i in text.split("Answer:")[-1].split(".") if len(i.split(" ")) > 3 and i.lower() != "Nothing follows"])
    # Some models continue the sequence generating  text like '### Instruction', the following line clean the generated text
    text = text.replace("Instructions", "").replace("#", "")
    text = text.rstrip(".:\n")
    return text
