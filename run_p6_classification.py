from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p6"
    tasks = {
        "algorithms" : ["random", "linspacer", "pcal", "mcuve", "bsnet", "bnc", "c1", "bsdr","bsdrattn","c1_wo_dsc","msobsdr"],
        "datasets": ["ghisaconus"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
