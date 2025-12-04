# @title Define plotting utils

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
import numpy as np
import matplotlib.pyplot as plt

from bokeh.models import Button
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column

output_notebook()


def plot_spikes(spikes, x_range=None, width=800, height=400):
    """
    Plots an IrregularTimeSeries object defined by spikes.timestamps and spikes.unit_index.

    Parameters:
    spikes: An object containing 'timestamps' and 'unit_index' attributes.
    """
    if x_range is None:
        x_range = (spikes.timestamps[0] * 1e3, spikes.timestamps[0] * 1e3 + 20_000)

    # Create a figure
    p = figure(
        x_axis_label="Time",
        y_axis_label="Unit Index",
        width=width,
        height=height,
        x_axis_type="datetime",
        x_range=x_range,
        title="Spikes",
    )

    # Prepare data for plotting
    x_values = spikes.timestamps * 1e3
    y_values = spikes.unit_index

    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(x=x_values, y=y_values))

    # Add scatter points to the plot
    p.scatter(
        "x",
        "y",
        source=source,
        size=5,
        color="navy",
        alpha=0.5,
        marker="dash",
        angle=np.pi / 2,
    )

    # Add range tool
    select = figure(
        height=height // 5,
        width=width,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
        x_axis_type="datetime",
        title="Average Population Activity",
    )
    select.xaxis.visible = False
    # select.yaxis.visible = False

    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    spike_times_int = spikes.timestamps.astype(int)
    population_activity = np.bincount(spike_times_int - spike_times_int[0])
    source = ColumnDataSource(
        data=dict(
            x=(np.arange(len(population_activity)) + spike_times_int[0]) * 1e3,
            y=population_activity,
        )
    )

    # select.line('x', 'y', source=source)
    # select.ygrid.grid_line_color = None
    # select.add_tools(range_tool)
    # p = column(p)

    return p


def plot_multi_time_series(
    data, field, x_range=None, y_axis_label=None, width=800, height=100
):
    # Create a figure
    if x_range is None:
        x_range = (data.timestamps[0] * 1e3, data.timestamps[0] * 1e3 + 10_000)

    if y_axis_label is None:
        y_axis_label = field

    p = figure(
        x_axis_label="Time",
        y_axis_label=y_axis_label,
        width=width,
        height=height,
        x_axis_type="datetime",
        x_range=x_range,
    )

    domain_start = data.domain.start * 1e3
    domain_end = data.domain.end * 1e3
    # Prepare data for plotting
    x_values = data.timestamps * 1e3
    y_values = getattr(data, field)

    x_values = np.concatenate([x_values, domain_start, domain_end])
    y_values = np.concatenate(
        [
            y_values,
            np.nan * np.ones((len(data.domain), *y_values.shape[1:])),
            np.nan * np.ones((len(data.domain), *y_values.shape[1:])),
        ]
    )

    # sort x_values and reorder y_values
    sort_indices = np.argsort(x_values)
    x_values = x_values[sort_indices]
    y_values = y_values[sort_indices]

    # If data is 1D then add a second dimension
    if y_values.ndim == 1:
        y_values = y_values[:, np.newaxis]

    n_dims = y_values.shape[1]
    # make a colour for each dim from viridis
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / n_dims) for i in range(n_dims)]
    colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b, a in colors
    ]
    for dim in range(n_dims):
        source = ColumnDataSource(data=dict(x=x_values, y=y_values[:, dim]))
        p.line(x="x", y="y", source=source, line_width=2, color=colors[dim])

    x_range = p.x_range

    p.xaxis.visible = False

    # Add range tool
    select = figure(
        height=height // 2,
        width=width,
        y_range=p.y_range,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
        x_axis_type="datetime",
    )
    select.xaxis.visible = False
    select.yaxis.visible = False

    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line("x", "y", source=source)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    p = column(select, p)

    # Add play controls
    # Add a button to control the range tool
    play_button = Button(label="Play", button_type="success")
    pause_button = Button(label="Pause", button_type="warning")
    speed_buttons = [Button(label=f"{2**i}x", button_type="primary") for i in range(5)]

    # Create a shared ColumnDataSource to store the interval ID
    shared_data = ColumnDataSource(data=dict(interval_id=[None], step_size=[1000]))

    # Update the CustomJS to use the shared data source
    play_button.js_on_click(
        CustomJS(
            args=dict(
                range_tool=range_tool, x_values=x_values, shared_data=shared_data
            ),
            code="""
        // Clear previous interval if it exists
        if (shared_data.data.interval_id[0]) {
            clearInterval(shared_data.data.interval_id[0]);
        }

        // Create new interval and store it in the shared data
        let new_interval = setInterval(() => {
            if (range_tool.x_range.end < x_values[x_values.length - 1]) {
                range_tool.x_range.start += shared_data.data.step_size[0] / 10;
                range_tool.x_range.end += shared_data.data.step_size[0] / 10;
            } else {
                clearInterval(shared_data.data.interval_id[0]);
                shared_data.data.interval_id[0] = null;
                shared_data.change.emit();
            }
        }, 100);

        shared_data.data.interval_id[0] = new_interval;
        shared_data.change.emit();
    """,
        )
    )
    pause_button.js_on_click(
        CustomJS(
            args=dict(shared_data=shared_data),
            code="""
        if (shared_data.data.interval_id[0]) {
            clearInterval(shared_data.data.interval_id[0]);
            shared_data.data.interval_id[0] = null;
            shared_data.change.emit();
        }
    """,
        )
    )
    for button in speed_buttons:
        button.js_on_click(
            CustomJS(
                args=dict(button=button, shared_data=shared_data),
                code="""
            shared_data.data.step_size[0] = 1000 * parseInt(button.label.replace('x', ''));
            shared_data.change.emit();
        """,
            )
        )

    # Add the buttons to the layout
    button_layout = row(play_button, pause_button, *speed_buttons)
    return p, x_range, button_layout


def make_plot(data, add_play_controls=False):
    p_latent, x_range, button_layout = plot_multi_time_series(
        data.latent,
        "pos",
        y_axis_label="latent",
    )

    p_spikes = plot_spikes(data.spikes, x_range=x_range, height=480)

    return column(
        button_layout,
        p_latent,
        p_spikes,
    )


def plot_dandi_data():
    from pynwb import NWBHDF5IO
    import numpy as np
    from temporaldata import (
        ArrayDict,
        IrregularTimeSeries,
        RegularTimeSeries,
        Interval,
        Data,
    )

    data_dir = "/Users/tomgeorge/Data/dandi"
    input_file = f"{data_dir}/sub-T_ses-CO-20130819_behavior+ecephys.nwb"
    io = NWBHDF5IO(input_file, "r")
    nwbfile = io.read()

    spike_train_list = nwbfile.units.spike_times_index[:]

    unit_ids = []
    unit_brain_areas = []
    for i in range(len(spike_train_list)):
        unit_ids.append(f"unit_{i}")
        unit_brain_areas.append(nwbfile.units.electrodes[i].location.item())

    spike_timestamps = np.array([])
    spike_unit_index = np.array([])

    for i in range(len(spike_train_list)):
        spike_train = spike_train_list[i]
        spike_timestamps = np.concatenate([spike_timestamps, spike_train])
        spike_unit_index = np.concatenate(
            [spike_unit_index, np.full_like(spike_train, fill_value=i)]
        )

    spikes = IrregularTimeSeries(
        timestamps=spike_timestamps,
        unit_index=spike_unit_index,
        domain="auto",
    )
    spikes.sort()

    timestamps = nwbfile.processing["behavior"]["Position"]["cursor_pos"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["cursor_pos"].data[:]
    cursor_vel = nwbfile.processing["behavior"]["Velocity"]["cursor_vel"].data[:]
    cursor_acc = nwbfile.processing["behavior"]["Acceleration"]["cursor_acc"].data[:]

    sampling_rate = 100  # Hz
    assert np.allclose(np.diff(timestamps), 1 / sampling_rate)

    cursor = RegularTimeSeries(
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        sampling_rate=sampling_rate,
        domain_start=timestamps[0],
        domain="auto",
    )

    trial_table = nwbfile.trials.to_dataframe().dropna()
    reach_intervals = Interval(
        start=trial_table.go_cue_time.values,
        end=trial_table.stop_time.values,
        result=trial_table.result.values,
        target_id=trial_table.target_id.values,
    )

    data = Data(
        spikes=spikes,
        latent=cursor,
        reach_intervals=reach_intervals,
        domain="auto",
    )

    p = make_plot(data, add_play_controls=True)
    return p
