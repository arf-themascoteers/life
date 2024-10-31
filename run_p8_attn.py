from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "p8"
    tasks = {
        "algorithms" : ["bsdrattn2","skipattn2"],
        "datasets": ["indian_pines","paviaU","salinas","ghisaconus","lucas_r"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
