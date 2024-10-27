from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "msobsdr_only"
    tasks = {
        "algorithms" : ["msobsdr"],
        "datasets": ["indian_pines"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True, test=False)
    summary, details = ev.evaluate()
