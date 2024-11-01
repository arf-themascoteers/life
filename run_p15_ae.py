from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p15"
    tasks = {
        "algorithms" : ["bsdr3ae2","msobsdr3ae"],
        "datasets": ["indian_pines","paviaU","salinas","ghisaconus","lucas_r"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
