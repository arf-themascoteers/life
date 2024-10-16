from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "1"
    tasks = {
        "algorithms" : ["pcal"],
        "datasets": ["indian_pines"],
        "target_sizes" : [5]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False)
    summary, details = ev.evaluate()
