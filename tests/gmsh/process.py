import sys
import os
import shutil
import subprocess
import yaml
import attr
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
import fracture
import repository_mesh
# import mesh_reading
import test_field
from bgem.gmsh import gmsh, field

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'bgem'))
from bgem.gmsh import heal_mesh


@attr.s(auto_attribs=True)


class ValueDescription:
    time: float
    position: str
    quantity: str
    unit: str


def substitute_placeholders(file_in, file_out, params):
    """
    Substitute for placeholders of format '<name>' from the dict 'params'.
    :param file_in: Template file.
    :param file_out: Values substituted.
    :param params: { 'name': value, ...}
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return used_params


def to_polar(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if z > 0:
        phi += np.pi
    return phi, rho


def plot_fr_orientation(fractures):
    family_dict = collections.defaultdict(list)
    for fr in fractures:
        x, y, z = fracture.FisherOrientation \
        .rotate(np.array([0, 0, 1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
        family_dict[fr.region].append([to_polar(z, y, x), to_polar(z, x, -y), to_polar(y, x, z)])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    for name, data in family_dict.items():
        # data shape = (N, 3, 2)
        data = np.array(data)
        for i, ax in enumerate(axes):
            phi = data[:, i, 0]
            r = data[:, i, 1]
            c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
    axes[0].set_title("X-view, Z-north")
    axes[1].set_title("Y-view, Z-north")
    axes[2].set_title("Z-view, Y-north")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
    fig.legend(loc = 1)
    fig.savefig("fracture_orientation.pdf")
    plt.close(fig)
    # plt.show()


def generate_fractures(config_dict):
    geom = config_dict["geometry"]
    fracture_box = geom["box_dimensions"]
    fr_box = geom["main_tunnel_length"]

    volume = np.product(fracture_box)
    pop = fracture.Population(volume)
    pop.initialize(geom["fracture_stats"])
    pop.set_sample_range([1, fr_box], max_sample_size=geom["n_frac_limit"])

    print("total mean size: ", pop.mean_size())

    pos_gen = fracture.UniformBoxPosition(fracture_box)
    fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    # fracture.fr_intersect(fractures)

    for fr in fractures:
        fr.region = "fr"
    used_families = set((f.region for f in fractures))
    for model in ["hm_params", "th_params", "th_params_ref"]:
        model_dict = config_dict[model]
        model_dict["fracture_regions"] = list(used_families)
        model_dict["boreholes_fracture_regions"] = [".{}_boreholes".format(f) for f in used_families]
        model_dict["main_tunnel_fracture_regions"] = [".{}_main_tunnel".format(f) for f in used_families]
    return fractures


def create_fractures_polygons(gmsh_geom, fractures):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    frac_obj = fracture.Fractures(fractures)
    frac_obj.snap_vertices_and_edges()
    shapes = []
    for fr, square in zip(fractures, frac_obj.squares):
        shape = gmsh_geom.make_polygon(square).set_region(fr.region)
        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def prepare_mesh(config_dict, fractures):
    mesh_name = config_dict["mesh_name"]
    mesh_file = mesh_name + ".msh"
    if not os.path.isfile(mesh_file):
        repository_mesh.make_mesh(config_dict, fractures, mesh_name, mesh_file)
        #test_field.apply_field4(config_dict, fractures, mesh_name, mesh_file)

    mesh_healed = mesh_name + "_healed.msh"
    if not os.path.isfile(mesh_healed):
        hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
        hm.heal_mesh(gamma_tol=0.01)
        hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
        hm.write()
        assert hm.healed_mesh_name == mesh_healed
    return mesh_healed


def check_conv_reasons(log_fname):
    with open(log_fname, "r") as f:
        for line in f:
            tokens = line.split(" ")
            try:
                i = tokens.index('convergence')
                if tokens[i + 1] == 'reason':
                    value = tokens[i + 2].rstrip(",")
                    conv_reason = int(value)
                    if conv_reason < 0:
                        print("Failed to converge: ", conv_reason)
                        return False
            except ValueError:
                continue
    return True


def call_flow(config_dict, param_key, result_files):
    # """
    # Redirect sstdout and sterr, return true on succesfull run.
    # :param arguments:
    # :return:
    # """

    params = config_dict[param_key]
    fname = params["in_file"]
    substitute_placeholders(fname + '_tmpl.yaml', fname + '.yaml', params)
    arguments = config_dict["_aux_flow_path"].copy()
    output_dir = "output_" + fname
    config_dict[param_key]["output_dir"] = output_dir
    if all([os.path.isfile(os.path.join(output_dir, f)) for f in result_files]):
        status = True
    else:
        arguments.extend(['--output_dir', output_dir, fname + ".yaml"])
        print("Running: ", " ".join(arguments))
        # print(fname + "_stdout")
        print(arguments)
        with open(fname + "_stdout", "w") as stdout:
            with open(fname + "_stderr", "w") as stderr:
                completed = subprocess.run(arguments, stdout=stdout, stderr=stderr)
                # print(completed)
        print("Exit status: ", completed.returncode)
        status = completed.returncode == 0
    conv_check = check_conv_reasons(os.path.join(output_dir, "flow123.0.log"))
    print("converged: ", conv_check)
    return status  # and conv_check


def prepare_th_input(config_dict):
    """
    Prepare FieldFE input file for the TH simulation.
    :param config_dict: Parsed config.yaml. see key comments there.
    """
    # pass
    # we have to read region names from the input mesh
    # input_mesh = gmsh_io.GmshIO(config_dict['hm_params']['mesh'])
    #
    # is_bc_region = {}
    # for name, (id, _) in input_mesh.physical.items():
    #     unquoted_name = name.strip("\"'")
    #     is_bc_region[id] = (unquoted_name[0] == '.')

    # read mesh and mechanichal output data
    mechanics_output = os.path.join(config_dict['hm_params']["output_dir"], 'mechanics.msh')
    ele_ids = np.array(list(mesh.elements.keys()), dtype=float)

    init_fr_cs = float(config_dict['hm_params']['fr_cross_section'])
    init_fr_K = float(config_dict['hm_params']['fr_conductivity'])
    init_bulk_K = float(config_dict['hm_params']['bulk_conductivity'])
    min_fr_cross_section = float(config_dict['th_params']['min_fr_cross_section'])
    max_fr_cross_section = float(config_dict['th_params']['max_fr_cross_section'])

    time_idx = 1
    time, field_cs = mesh.element_data['cross_section_updated'][time_idx]

    # cut small and large values of cross-section
    cs = np.maximum(np.array([v[0] for v in field_cs.values()]), min_fr_cross_section)
    cs = np.minimum(cs, max_fr_cross_section)

    K = np.where(
        cs == 1.0,      # condition
        init_bulk_K,    # true array
        init_fr_K * (cs / init_fr_cs) ** 2
    )

    # get cs and K on fracture elements only
    fr_indices = np.array([int(key) for key, val in field_cs.items() if val[0] != 1])
    cs_fr = np.array([cs[i] for i in fr_indices])
    k_fr = np.array([K[i] for i in fr_indices])

    # compute cs and K statistics and write it to a file
    fr_param = {}
    avg = float(np.average(cs_fr))
    median = float(np.median(cs_fr))
    interquantile = float(1.5 * (np.quantile(cs_fr, 0.75) - np.quantile(cs_fr, 0.25)))
    fr_param["fr_cross_section"] = {"avg": avg, "median": median, "interquantile": interquantile}

    avg = float(np.average(k_fr))
    median = float(np.median(k_fr))
    interquantile = float(1.5 * (np.quantile(k_fr, 0.75) - np.quantile(k_fr, 0.25)))
    fr_param["fr_conductivity"] = {"avg": avg, "median": median, "interquantile": interquantile}

    with open('fr_param_output.yaml', 'w') as outfile:
        yaml.dump(fr_param, outfile, default_flow_style=False)

    # mesh.write_fields('output_hm/th_input.msh', ele_ids, {'conductivity': K})
    th_input_file = 'th_input.msh'
    with open(th_input_file, "w") as fout:
        mesh.write_ascii(fout)
        mesh.write_element_data(fout, ele_ids, 'conductivity', K[:, None])
        mesh.write_element_data(fout, ele_ids, 'cross_section_updated', cs[:, None])

    # create field for K (copy cs)
    # posun dat K do casu 0
    # read original K = oK (define in config yaml)
    # read original cs = ocs (define in config yaml)
    # compute K = oK * (cs/ocs)^2
    # write K

    # posun dat cs do casu 0
    # write cs

    # mesh.element_data.


def get_result_description():
    """
    :return:
    """
    end_time = 30
    values = [[ValueDescription(time=t, position="extraction_well", quantity="power", unit="MW"),
                ValueDescription(time=t, position="extraction_well", quantity="temperature", unit="Celsius deg.")
                ] for t in np.linspace(0, end_time, 0.1)]
    power_series, temp_series = zip(*values)
    return power_series + temp_series


def extract_time_series(yaml_stream, regions, extract):
    """
    :param yaml_stream:
    :param regions:
    :return: times list, list: for every region the array of value series
    """
    data = yaml.safe_load(yaml_stream)['data']
    times = set()
    reg_series = {reg: [] for reg in regions}

    for time_data in data:
        region = time_data['region']
        if region in reg_series:
            times.add(time_data['time'])
            power_in_time = extract(time_data)
            reg_series[region].append(power_in_time)
    times = list(times)
    times.sort()
    series = [np.array(region_series, dtype=float) for region_series in reg_series.values()]
    return np.array(times), series


def extract_results(config_dict):
    """
    :param config_dict: Parsed config.yaml. see key comments there.
    : return
    """
    # bc_regions = ['.fr_left_well', '.left_well', '.fr_right_well', '.right_well']
    bc_regions = ['.fr_boreholes', '.boreholes', '.fr_main_tunnel', '.main_tunnel']
    out_regions = bc_regions[2:]
    output_dir = config_dict["th_params"]["output_dir"]
    with open(os.path.join(output_dir, "energy_balance.yaml"), "r") as g:
        power_times, reg_powers = extract_time_series(g, bc_regions, extract=lambda frame: frame['data'][0])
        power_series = -sum(reg_powers)

    with open(os.path.join(output_dir, "Heat_AdvectionDiffusion_region_stat.yaml"), "r") as h:
        temp_times, reg_temps = extract_time_series(h, out_regions, extract=lambda frame: frame['average'][0])
    with open(os.path.join(output_dir, "water_balance.yaml"), "r") as h:
        flux_times, reg_fluxes = extract_time_series(h, out_regions, extract=lambda frame: frame['data'][0])
    sum_flux = sum(reg_fluxes)
    avg_temp = sum([temp * flux for temp, flux in zip(reg_temps, reg_fluxes)]) / sum_flux
    print("temp: ", avg_temp)
    return temp_times, avg_temp, power_times, power_series


def plot_exchanger_evolution(temp_times, avg_temp, power_times, power_series):
    abs_zero_temp = 273.15
    year_sec = 60 * 60 * 24 * 365

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    temp_color = 'red'
    ax1.set_xlabel('time [y]')
    ax1.set_ylabel('Temperature [C deg]', color=temp_color)
    ax1.plot(temp_times[1:] / year_sec, avg_temp[1:] - abs_zero_temp, color=temp_color)
    ax1.tick_params(axis='y', labelcolor=temp_color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    pow_color = 'blue'
    ax2.set_ylabel('Power [MW]', color=pow_color)  # we already handled the x-label with ax1
    ax2.plot(power_times[1:] / year_sec, power_series[1:] / 1e6, color=pow_color)
    ax2.tick_params(axis='y', labelcolor=pow_color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def sample_mesh_repository(mesh_repository):
    mesh_file = np.random.choice(os.listdir(mesh_repository))
    healed_mesh = "random_fractures_healed.msh"
    shutil.copyfile(os.path.join(mesh_repository, mesh_file), healed_mesh)
    heal_ref_report = {'flow_stats': {'bad_el_tol': 0.01, 'bad_elements': [], 'bins': [], 'hist': []},
                       'gamma_stats': {'bad_el_tol': 0.01, 'bad_elements': [], 'bins': [], 'hist': []}}
    with open("random_fractures_heal_stats.yaml", "w") as f:
        yaml.dump(heal_ref_report, f)
    return healed_mesh


def setup_dir(config_dict, clean=False):
    for g in config_dict["copy_files"]:
        shutil.copyfile(os.path.join(script_dir, g), os.path.join(".", g))
    flow_exec = config_dict["flow_executable"].copy()
    # if not os.path.isabs(flow_exec[0]):
    #     flow_exec[0] = os.path.join(script_dir, flow_exec[0])
    config_dict["_aux_flow_path"] = flow_exec


def sample(config_dict):

    setup_dir(config_dict, clean=True)
    mesh_repo = config_dict.get('mesh_repository', None)
    fractures = generate_fractures(config_dict)
    if mesh_repo:
        healed_mesh = sample_mesh_repository(mesh_repo)
    else:
        # plot_fr_orientation(fractures)
        healed_mesh = prepare_mesh(config_dict, fractures)

    healed_mesh_bn = os.path.basename(healed_mesh)
    config_dict["hm_params"]["mesh"] = healed_mesh_bn
    config_dict["th_params"]["mesh"] = healed_mesh_bn
    config_dict["th_params_ref"]["mesh"] = healed_mesh_bn

    # hm_succeed = call_flow(config_dict, 'hm_params', result_files=["mechanics.msh"])

    # th_succeed = call_flow(config_dict, 'th_params_ref', result_files=["energy_balance.yaml"])

    # print("th_succeed = " + str(th_succeed))

    # th_succeed = False
    # if hm_succeed:
    #     prepare_th_input(config_dict)
    #     th_succeed = call_flow(config_dict, 'th_params', result_files=["energy_balance.yaml"])
    #
    #     if th_succeed:
    #        series = extract_results(config_dict)
    #        plot_exchanger_evolution(*series)

    # if th_succeed:
    #     series = extract_results(config_dict)
    #     plot_exchanger_evolution(*series)
    # print("Finished")


if __name__ == "__main__":

    sample_dir = "output"

    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(script_dir, "config.yaml"), "r") as f:
         config_dict = yaml.safe_load(f)

    os.chdir(sample_dir)

    # np.random.seed()
    # seed_number = np.random.randint(2**20)
    # print(seed_number)
    # np.random.seed(seed_number)

    # f = open("seed_numbers.txt", "w")
    # f.write(str(seed_number) + "\n")
    # f.close()

    np.random.seed(878870)

    ###############################################################################################
    # fieldy nefunguji, spadne pri meshovani
    # v adresari otput jen ukazka funkcniho meshe pro field(const)
    '''
    fractures = generate_fractures(config_dict)
    test_field.apply_field_pukliny(config_dict, fractures, dim=3, tolerance=0.3, max_mismatch=10, mesh_name="geom_pukliny")
    '''
    ###############################################################################################

    ###############################################################################################
    # geometrie bez puklin, funguje vcetne fieldu
    # parametry meshe (fieldu) zadane zatim primo uvnitr funkce
    test_field.apply_field7(config_dict, dim=3, tolerance=0.3, max_mismatch=10, mesh_name="geom_komplet")
    ###############################################################################################

    # sample(config_dict)

    # shifting the geometry, marking edz physical group
    # mesh_reading.shift_mesh(config_dict)

