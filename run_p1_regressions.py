from task_runner import TaskRunner

#All algorithms for lucas except for multi-bsdr
if __name__ == '__main__':
    tag = "p1"
    tasks = {
        "algorithms" : ["random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr","bsdrattn","c1_wo_dsc"],
        "datasets": ["lucas_r"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
