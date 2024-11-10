from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p21"
    tasks = {
        "algorithms" : ["bsnet", "bnc", "c1"],
        "datasets": ["ghisaconus_health"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
