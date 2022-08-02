import sys


def print_host_results(rb, depth_limit=sys.maxsize, truncate_count=100, depth=0):
    t = rb["runtime"]
    if depth == 0:
        print(f"Two iteration runtime: {t:>20} (in us, same below)")

    module_perc_sum = 0
    dominated_perc_sum = 0
    dominated_count = 0
    space_padding = " " * (depth + 1) * 5
    for k, v in sorted(rb["subs"].items(), key=lambda x: x[1]["runtime"], reverse=True):
        vt = str(v["runtime"])
        perc = v["runtime"] / t
        module_perc_sum += perc
        # Truncate results for brevity
        if dominated_count < truncate_count:
            dominated_perc_sum += perc
            count = 0
            for kk, vv in v["stats"].items():
                count += vv["count"]
            print(f"{space_padding}{(k+':'):<40} {('(' + vt):>{(depth+2) * 5}}, {(perc * 100):.2f}%, {count})")

            # DFS and print
            if depth < depth_limit and len(v["subs"].keys()) != 0:
                print_host_results(v, depth_limit, truncate_count, depth=depth+1)
            dominated_count += 1

    # If there's still remaining time, print it under "Others"
    if abs(module_perc_sum - dominated_perc_sum) > 1e-4:
        other_time = "{:.1f}".format((module_perc_sum - dominated_perc_sum) * t)
        print(f"{space_padding}{'Others:':<40} {('(' + other_time):>{(depth+2) * 5}}, {((module_perc_sum - dominated_perc_sum) * 100):.2f}%)")

    # Unaccounted time
    unaccounted_time = "{:.1f}".format((1 - module_perc_sum) * t)
    print(f"{space_padding}{'Unaccounted:':<40} {('(' + unaccounted_time):>{(depth+2) * 5}}, {((1 - module_perc_sum) * 100):.2f}%)")


def get_host_runtime_breakdown(events, total_time):
    runtime_breakdown = {}

    def construct_rb(events, rb):
        for e in events:
            name = e.name()
            shape = (e.input_shape(), e.output_shape())
            ex_id = e.external_id()
            if name not in rb.keys():
                rb[name] = {}
                rb[name]["subs"] = {} # All the children
                rb[name]["runtime"] = 0.0 # Event runtime seen on the host
                rb[name]["stats"] = {}
            rb[name]["runtime"] += e.duration()
            
            if shape not in rb[name]["stats"].keys():
                rb[name]["stats"][shape] = {}
                rb[name]["stats"][shape]["count"] = 0
            rb[name]["stats"][shape]["count"] += 1

            # DFS for children
            construct_rb(e.children, rb[name]["subs"])

    construct_rb(events, runtime_breakdown)
    runtime_breakdown = {
        "runtime": total_time,
        "subs": runtime_breakdown
    }

    return runtime_breakdown
