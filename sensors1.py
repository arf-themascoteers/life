from task_runner import TaskRunner


if __name__ == '__main__':
    tag = "s1"
    tasks = {
        "algorithms" : ["mcuve","spa","bsnet","bnc","bsdr"],
        "datasets": ["ghisaconus"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
