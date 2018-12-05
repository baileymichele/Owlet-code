import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

from bokeh.models import Legend
from bokeh.plotting import figure
from bokeh.io import show, curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models.widgets import TextInput, DatePicker, Button
from bokeh.models import DatetimeTickFormatter, Tool, String, CustomJS
from bokeh.models import ColumnDataSource, HoverTool, CrosshairTool, BoxZoomTool, ZoomOutTool


JS_CODE_SAVE = """
import * as p from "core/properties"
import {ActionTool, ActionToolView} from "models/tools/actions/action_tool"

export class NewSaveView extends ActionToolView

  # this is executed when the button is clicked
  doit: () ->
    @plot_view.save(@model.save_name)

export class CustomSaveTool extends ActionTool
  default_view: NewSaveView
  type: "CustomSaveTool"

  tool_name: "Save"
  icon: "bk-tool-icon-save"

  @define { save_name: [ p.String ] } """

class CustomSaveTool(Tool):
    """
    Save a plot with a custom name.
    Usage: CustomSaveTool(save_name = name)
    """
    __implementation__ = JS_CODE_SAVE
    save_name = String()

def binary(x):
    bi = list(bin(x)[2:])[::-1]
    indices = np.where(np.array(bi) == "1")[0]
    return indices

def map_indices(x):
    indices = binary(x)
    notifications_mask = {0:"sock off",
                          1:"movement",
                          2:"buffer reset",
                          3:"HR out of range",
                          4:"signal saturating DC amplifier",
                          5:"signal saturating DC amplifier",
                          6:"signal saturating DC amplifier",
                          7:"signal saturating DC amplifier",
                          8:"signal saturating DC amplifier",
                          9:"signal saturating DC amplifier",
                          10:"DC signal low",
                          11:"LED adjusting",
                          12:"low pulse amplitude",
                          13:"AC Ambient Signal is Too High",
                          14:"O2 out of range",
                          15:"Invalid HR"}
    notifications = []
    for i in range(len(indices)):
        notifications.append(notifications_mask[indices[i]])
    if len(notifications) == 0:
        return ["Good"]
    return list(set(notifications))

def day_limits(df_full):
    return df_full.timestamp.min(), df_full.timestamp.max()

def load_data(filename):
    df = pd.read_csv(filename)
    return df#[df.base_state > 3]

def dsn_date(df_full, dsn, dates):

    df = df_full[df_full.dsn == dsn]
    previous_day = str(df.timestamp.max() - timedelta(days=1))
    if not dates:
        dates = [previous_day, str(df.timestamp.max())]
    df = df[df.timestamp > dates[0]]
    df = df[df.timestamp < dates[1]]
    df = df.sort_values(by=['timestamp'])
    return df

def get_source(df, names):
    '''Put all data needed for plotting into a dictionary'''

    def datetime(x):
        '''Function to format the datetime data'''
        return np.array(x, dtype=np.datetime64)

    maxs = names.copy()
    maxs.remove("movement_raw")
    maxs.remove("skin_temp")

    max_hr = df[maxs].max().max()
    min_val = df[maxs].min().min()
    height = 12*(len(names)+1)
    horizontal_points = [max_hr + height]*df.shape[0]

    good_data = data_to_plot(df[df.notifications_mask == 0], names[:3])
    bad_data = data_to_plot(df[df.notifications_mask != 0], names[:3])

    bm = ["xs", "base_state", "movement_raw"]
    base_mvmt = data_to_plot(df[df.base_state > 3], bm[1:])
    # Where data is stored so it can be displayed in the tooltip
    source_data = {
        'horizontal'         : horizontal_points,
        'date'               : datetime(df['timestamp']),
        'notifications_mask' : df.notifications_mask #.apply(map_indices),#.apply(notifications),
    }
    source_data.update({names[i] : df[names[i]] for i in range(len(names))})
    source_data["skin_temp"] = df["skin_temp"]/2
    source_data1 = {'plot_' + bm[i] : base_mvmt[i] for i in range(len(bm))}

    source_data2 = {'good_x' : good_data[0]}
    source_data2.update({'good_' + names[i] : good_data[i+1] for i in range(3)})
    source_data3 = {'bad_x' : bad_data[0]}
    source_data3.update({'bad_' + names[i] : bad_data[i+1] for i in range(3)})

    nine = start_indices(df, 9)
    ten = start_indices(df, 10)
    yellow = np.insert(ten, 0, nine) # combine indices for 9s and 10s
    red = start_indices(df, 12)

    source_data4 = {'yellow' : [[df.iloc[i].timestamp]*20 for i in yellow]}
    source_data4.update({'ys' : [np.linspace(0,500,20) for _ in yellow]})

    source_data5 = {'red' : [[df.iloc[i].timestamp]*20 for i in red]}
    source_data5.update({'red_ys' : [np.linspace(0,500,20) for _ in red]})

    return source_data, source_data1, source_data2, source_data3, source_data4, source_data5, min_val, max_hr

def start_indices(df, x):
    "Returns the indices where the base state changes to x"
    value = df.base_state.eq(x)
    indices = np.ndarray.flatten(np.argwhere(value))
    differences = np.diff(indices)
    if len(indices) != 0:
        differences = np.insert(differences, 0, 0)
    final_indices = list(np.ndarray.flatten(np.argwhere(differences != 1)))
    return indices[final_indices]

def data_to_plot(df, save):
    data = [[] for i in range(len(save)+1)]
    remove = 0
    d = pd.Timedelta(2, 's')
    while df.shape[0] != 0:

        consecutive = df.timestamp.diff().fillna(0).abs().le(d)

        idx_loc = df.index.get_loc(consecutive.idxmin())
        if idx_loc == 0:
            df_to_plot = df
            df = df.iloc[0:0] # empty the df
        else:
            df_to_plot = df.iloc[:idx_loc]
            remove = df_to_plot.shape[0]
            df = df.iloc[remove:]

        data[0].append(df_to_plot.timestamp)
        for i in range(len(save)):
            if save[i] == "skin_temp":
                data[i+1].append(df_to_plot[save[i]]/2)
            elif save[i] == "base_state":
                data[i+1].append(df_to_plot[save[i]]*10)
            elif save[i] == "movement_raw":
                data[i+1].append(df_to_plot[save[i]]/4)
            else:
                data[i+1].append(df_to_plot[save[i]])

    return data

def plot_data(df, dsn):
    '''Plot the data in the given dataframe

    Parameters:
        df        (dataframe) - data from the given dsn
        dsn       (str) - device identifier

    '''

    names = ["heart_rate_avg", "oxygen_avg", "skin_temp", "base_state", "movement_raw"]
    colors = ["blue", "orange", "red", "purple", "green"]
    alphas = [.8,8,.8,.7,.6]

    # Where data is stored so it can be displayed in the tooltip
    source_data, source_data1, source_data2, source_data3, source_data4, source_data5, min_val, max_hr = get_source(df, names)

    source = ColumnDataSource(data=source_data)
    source1 = ColumnDataSource(data=source_data1)
    source2 = ColumnDataSource(data=source_data2)
    source3 = ColumnDataSource(data=source_data3)
    source4 = ColumnDataSource(data=source_data4)
    source5 = ColumnDataSource(data=source_data5)

    hover_tool = HoverTool(
        tooltips=[
        ( 'time',   '@date{%T}' ),] +
        [(name, '@{}'.format(name)) for name in names] +
        [('notification', '@notifications_mask')], #data: good; corrupt


        formatters={'date'  : 'datetime',}, # use default 'numeral' formatter for other fields
        mode='vline', renderers=[]
    )

    # Vertical line where the mouse is
    crosshair = CrosshairTool(dimensions='height', line_alpha=.6)

    box_zoom = BoxZoomTool(dimensions='width')
    zoom_out = ZoomOutTool(dimensions='width', factor=.5)
    save = CustomSaveTool(save_name=dsn)

    # Create figure; use webgl for better rendering
    tools=[save, box_zoom, 'xpan', zoom_out, 'reset', hover_tool, crosshair]
    p = figure(width=950, height=500, title="{} Data".format(dsn), x_axis_type="datetime",
           tools=tools, toolbar_location="above", y_range=(0,max_hr+(24*(len(names)+1))), output_backend='webgl')

    # To have the Legend outside the plot, each line needs to be added to it
    legend_it = []

    for i in range(len(names)):
        legend_line = p.line(x=df.timestamp.iloc[-1:], y=0, color=colors[i], alpha=1,line_width=2)
        if names[i] in ["movement_raw", "base_state"]:
            legend_it.append((names[i], [legend_line, p.multi_line(xs='plot_xs', ys='plot_'+names[i], color=colors[i], alpha=alphas[i], source=source1)]))
        else:
            bad_data_line = p.multi_line(xs='bad_x', ys='bad_'+names[i], color=colors[i], alpha=.1, source=source3)
            legend_it.append((names[i], [legend_line, p.multi_line(xs='good_x', ys='good_'+names[i], color=colors[i], alpha=alphas[i], source=source2), bad_data_line]))

    legend_it.append(("Yellow notifications", [p.multi_line(xs='yellow', ys='ys', color='#F5BE41', line_dash='dashed', source=source4)]))
    legend_it.append(("Red notifications", [p.multi_line(xs='red', ys='red_ys', color='red', line_dash='dashed', source=source5)]))

    # Creating a location for the tooltip box to appear (so it doesn't cover the data)
    horizontal_line = p.line(x='date', y='horizontal', color='white', alpha =0, source=source)
    hover_tool.renderers.append(horizontal_line)

    p.xaxis.axis_label = 'Time'
    p.xaxis.formatter=DatetimeTickFormatter(
            days=["%m/%d %T"],
            months=["%m/%d %T"],
            hours=["%m/%d %T"],
            minutes=["%m/%d %T"],
            seconds=["%m/%d %T"],
            minsec=["%m/%d %T"],
            hourmin=["%m/%d %T"]
    )

    legend = Legend(items=legend_it)
    legend.click_policy="hide"    # Hide lines when they are clicked on
    p.add_layout(legend, 'right')
    return p, source, source1, source2, source3, source4, source5

def update_data(p, text_title, text_save, source, source1, source2, source3, source4, source5, dsn, df_full, dates=None):
    names = ['heart_rate_avg', 'oxygen_avg', 'skin_temp', 'base_state', 'movement_raw']
    df = dsn_date(df_full, dsn, dates)
    if df.shape[0] == 0:
        return df
    source.data, source1.data, source2.data, source3.data, source4.data, source5.data, min_val, max_hr = get_source(df, names)

    # Title/save update dsn
    text_title.value = dsn + " Data"
    text_save.value = dsn

    # change y axis range:
    p.y_range.start = max(0,min_val-10)
    p.y_range.end = max_hr+(24*(len(names)+1))
    # return df? to change calendar to match dsn
    return df

def set_up_widgets(p, source, source1, source2, source3, source4, source5, df, df_full, text_dsn):
    dsn = text_dsn.value
    # Set up widgets
    text_title = TextInput(title="Title:", value="{} Data".format(dsn))
    text_save = TextInput(title="Save As:", value=dsn)

    max_for_dsn = df.timestamp.max()
    min_day, max_day = day_limits(df_full)
    calendar = DatePicker(title="Day:", value=date(max_for_dsn.year, max_for_dsn.month, max_for_dsn.day+1),
            max_date=date(max_day.year, max_day.month, max_day.day+1), min_date=date(min_day.year, min_day.month, min_day.day+1))
    button = Button(label="Update", button_type="success")

    # Set up callbacks
    def update_title(attrname, old, new):
        p.title.text = text_title.value

    def update_save(attrname, old, new):
        p.tools[0].save_name = text_save.value

    def update():
        text_dsn.value = text_dsn.value.strip(" ")  # Get rid of extra space
        # Make sure time is valid
        date_start = "{} 00:00:00".format(calendar.value)
        date_end = "{} 23:59:59".format(calendar.value)

        df_new = update_data(p, text_title, text_save, source, source1, source2, source3, source4, source5, text_dsn.value, df_full,dates=[date_start, date_end])
        # if df_new is going to be used, make sure it's not empty:
        # if df_new is empty...

    text_title.on_change('value', update_title)
    text_save.on_change('value', update_save)

    button.on_click(update)
    button.js_on_click(CustomJS(args=dict(p=p), code="""p.reset.emit()"""))

    # Set up layouts and add to document
    inputs = widgetbox(text_title, text_save, calendar, button)

    curdoc().add_root(row(inputs, p, width=1300))



# df_full = pd.read_csv("two_reds/AC000W000338438_2017_0220_0223.csv", header=None)
# with open("two_reds/column_names.txt") as f:
#     content = f.readlines()
# #     print(content)
# content = content[0].split(',')
# df_full.columns = content
# skin_temp is skin_temperature; notifications_mask no s...

df_full = load_data("yellow2.csv")
df_full.timestamp = pd.to_datetime(df_full.timestamp, unit='s')
text_dsn = TextInput(title="DSN:")
dsn_button = Button(label="Update DSN", button_type="success")
curdoc().add_root(row(text_dsn))
def dsn_text_update(attrname, old, new):
    df = dsn_date(df_full, text_dsn.value, None)
    # df must not be empty
    p, source, source1, source2, source3, source4, source5 = plot_data(df, text_dsn.value)
    text_dsn.remove_on_change('value', dsn_text_update)
    set_up_widgets(p, source, source1, source2, source3, source4, source5, df, df_full, text_dsn)

text_dsn.on_change('value', dsn_text_update)
