from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "bs"
    tasks = {
        "algorithms" : ["bnc","bsnet"],
        "datasets": ["indian_pines","salinas","paviaU","ghisaconus","lucas_texture_r"],
        "target_sizes" : [5,10,15,20,25,30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=False, verbose=False, test=False)
    summary, details = ev.evaluate()
