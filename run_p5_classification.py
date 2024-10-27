from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p5"
    tasks = {
        "algorithms" : ["random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr","c1_wo_dsc","msobsdr"],
        "datasets": ["salinas"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
