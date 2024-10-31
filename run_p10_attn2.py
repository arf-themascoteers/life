from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p10"
    tasks = {
        "algorithms" : [],
        "props": [],
        "datasets": ["indian_pines"],
        "target_sizes" : [10]
    }
    for i in range(10,101):
        tasks["algorithms"].append("bsdrattn2spec")
        tasks["props"].append({"shortlist":i})
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
