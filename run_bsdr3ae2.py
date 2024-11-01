from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "bsdr3ae2"
    tasks = {
        "algorithms" : ["bsdr3ae2"],
        "datasets": ["indian_pines"],
        "target_sizes" : list(range(10,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False)
    summary, details = ev.evaluate()
