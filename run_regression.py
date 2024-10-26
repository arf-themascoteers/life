from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "r1"
    tasks = {
        #"algorithms" : ["random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr"],
        "algorithms" : ["bnc", "c1", "bsdr", "bsdrattn"],
        "datasets": ["lucas_r"],
        "target_sizes" : [5,10,15,20,25,30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
