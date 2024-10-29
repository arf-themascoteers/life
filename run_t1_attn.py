from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "t1"
    tasks = {
        "algorithms" : ["skipattn","bsdrattn","lateattn","stage3attn"],
        "datasets": ["indian_pines","paviaU","salinas","ghisaconus","lucas_r"],
        "target_sizes" : list(range(2,31))
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=False, test=False, split="custom")
    summary, details = ev.evaluate()
