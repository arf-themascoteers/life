from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "r1"
    tasks = {
        #"algorithms" : ["random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr"],
        "algorithms" : ["bsdr"],
        "datasets": ["lucas_r"],
        "target_sizes" : [30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True, test=False)
    summary, details = ev.evaluate()
