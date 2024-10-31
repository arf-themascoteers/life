from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "m1"
    tasks = {
        "algorithms" : ["bsdr","c1","bsdrattn2","skipattn2"],
        "datasets": ["indian_pines","paviaU","salinas","ghisaconus","lucas_r"],
        "target_sizes" : list(range(2,151))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
