import sys

def print_major_device_results(device_runtime, drb, flatten, truncate_count=100, depth=0):
    t = drb["runtime"]
    t_perc = t / device_runtime * 100.0
    if t == 0.0:
        return
    idle = device_runtime - t
    idle_perc = idle / device_runtime * 100.0
    
    if depth == 0:
        print(f"    Total device time: {(device_runtime)} (in us, same below)")
        print(f"    Device idle time: {idle} ({idle_perc:.2f}%)")
        print(f"    Device active time: {t} ({t_perc:.2f}%)")
        
    module_perc_sum = 0
    dominated_perc_sum = 0
    dominated_count = 0
    space_padding = " " * ((depth + 1) * 2 + 4)
    
    for key, v in sorted(drb.items(), key=lambda x: x[1]["runtime"] if (isinstance(x[1], dict) and "runtime" in x[1].keys()) else -1, reverse=True):
        if key == "runtime":
            continue
        vt = str(v["runtime"])
        perc = v["runtime"] / t
        module_perc_sum += perc
        count = v["count"]
        # Truncate results for brevity
        if dominated_count < truncate_count:
            dominated_perc_sum += perc
            name = key[0]
            shape = str(key[1])
            if shape == '()' or shape == "((),)":
                print(f"{space_padding}{(name+':'):<52} {('(' + vt):>{(depth+2) * 3}}, {(perc * 100):.2f}%, {count})")
            else:
                print(f"{space_padding}{(name+':'):<52} {('(' + vt):>{(depth+2) * 3}}, {(perc * 100):.2f}%, {count}) {(shape):>60}")
                
            # DFS and print
            if "occ" in v.keys(): # Only ops and modules have 'occ', saving all occurences of this op or module with roughly the same runtime
                print_major_device_results(device_runtime, v["occ"][0], flatten, parent_name=key, truncate_count=truncate_count, depth=depth+1)
            dominated_count += 1

    # If there's still remaining time, print it under "Others"
    if abs(module_perc_sum - dominated_perc_sum) > 1e-4:
        other_time = "{:.1f}".format((module_perc_sum - dominated_perc_sum) * t)
        print(f"{space_padding}{'Others:':<52} {('(' + other_time):>{(depth+2) * 5}}, {((module_perc_sum - dominated_perc_sum) * 100):.2f}%)")
        
    # Unaccounted time
    if abs(1 - module_perc_sum) > 1e-4:
        unaccounted_time = "{:.1f}".format((1 - module_perc_sum) * t)
        print(f"{space_padding}{'Unaccounted:':<52} {('(' + unaccounted_time):>{(depth+2) * 5}}, {((1 - module_perc_sum) * 100):.2f}%)")


def get_major_device_results(device_runtime, drb, flatten, parent_name="total"):
    t = drb["runtime"]
    if t == 0.0:
        return
    idle = device_runtime - t
        
    module_perc_sum = 0
    dominated_perc_sum = 0
    
    if parent_name not in flatten.keys():
        flatten[parent_name] = {}
        flatten[parent_name]["runtime"] = 0.0
        flatten[parent_name]["subs"] = {}
    flatten[parent_name]["runtime"] += t
    
    for key, v in sorted(drb.items(), key=lambda x: x[1]["runtime"] if (isinstance(x[1], dict) and "runtime" in x[1].keys()) else -1, reverse=True):
        if key == "runtime":
            continue
        perc = v["runtime"] / t
        module_perc_sum += perc
        dominated_perc_sum += perc
            
        # DFS and print
        if "occ" in v.keys(): # Only ops and modules have 'occ', saving all occurences of this op or module with roughly the same runtime
            get_major_device_results(device_runtime, v["occ"][0], flatten, parent_name=key)

        if parent_name not in flatten:
            flatten[parent_name] = {}
            flatten[parent_name]["subs"] = {}
            flatten[parent_name][key] = {}
        if key not in flatten[parent_name]["subs"].keys():
            flatten[parent_name]["subs"][key] = 0.0
        flatten[parent_name]["subs"][key] += v["runtime"]

    # If there's still remaining time, print it under "Others"
    if abs(module_perc_sum - dominated_perc_sum) > 1e-4:
        other_time = "{:.1f}".format((module_perc_sum - dominated_perc_sum) * t)
        if "others" not in flatten[parent_name]["subs"].keys():
            flatten[parent_name]["subs"]["others"] = 0.0
        flatten[parent_name]["subs"]["others"] += float(other_time)
        
    # Unaccounted time
    if abs(1 - module_perc_sum) > 1e-4:
        unaccounted_time = "{:.1f}".format((1 - module_perc_sum) * t)
        if "unaccounted" not in flatten[parent_name]["subs"].keys():
            flatten[parent_name]["subs"]["unaccounted"] = 0.0
        flatten[parent_name]["subs"]["unaccounted"] += float(unaccounted_time)


def print_all_device_results(roots, odr, depth=0):
    space_padding = " " * depth * 4
    tmp_space_padding = " " * (depth + 1) * 4
    for r in roots:
        ex_id = r.external_id()
        if r.has_device_calls:
            print(f"{space_padding}{r.name()}")
            if ex_id in odr.keys():
                result = odr[ex_id]
                for _, d in result.items():
                    for key, v in d.items():
                        kernel_name = key[0]
                        shape = key[1]
                        kernel_count = v["count"]
                        kernel_time = v["runtime"]
                        print(f"{tmp_space_padding}{(kernel_name+':'):<44} {('( ' + str(shape)):>{(depth+2) * 4}}, {kernel_count}, {kernel_time} )")
            else:
                print_all_device_results(r.children, odr, -1, depth=depth+1)


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
