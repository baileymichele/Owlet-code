import numpy as np
import pandas as pd
from datetime import date, datetime
from sqlalchemy import create_engine

from bokeh.models import Legend
from bokeh.plotting import figure
from bokeh.layouts import row, widgetbox
from bokeh.io import output_notebook, show, curdoc
from bokeh.models import DatetimeTickFormatter, Tool, String, CustomJS
from bokeh.models.widgets import TextInput, DateRangeSlider, DatePicker, RangeSlider, Button
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

# Function to clarify notifications mask
def notifications(x):
    return "Data is good" if x == 0 else "Data may be corrupted"


def load_data(table="day_data", dsn_num="AC000W002577443", dates=None):
    '''Query for the data of the given dsn. If dates are given, find data between those dates'''

    engine = create_engine("postgresql+psycopg2://postgres:owlet@localhost:5433/smart_sock")
    conn = engine.raw_connection()
    df = pd.read_sql("select * from {} where dsn='{}' order by row_timestamp".format(table, dsn_num), conn)
    if dates:
        df = dsn_date(df, dates[0], dates[1])
    return df

def dsn_date(df, date_start, date_end):
    '''Select only data after date_start and before date_end'''

    df = df[df.row_timestamp > date_start]
    df = df[df.row_timestamp < date_end]
    df = df.sort_values(by=['row_timestamp'])
    return df

def get_source(df, y_values, names, circles):
    '''Put all data needed for plotting into a dictionary'''

    def datetime(x):
        '''Function to format the datetime data'''
        return np.array(x, dtype=np.datetime64)

    # Valid data points
    df_good = df[df.notifications_mask == 0]

    maxs = names.copy()
    maxs.remove("movement_raw")

    max_hr = df[maxs].max().max()
    min_val = df[maxs].min().min()
    height = 12*(len(names)+1)
    horizontal_points = [max_hr + height]*df.shape[0]

    # Where data is stored so it can be displayed in the tooltip
    source_data = {
        'horizontal'         : horizontal_points,
        'date'               : datetime(df['row_timestamp']),
        'notifications_mask' : df.notifications_mask#.apply(map_indices),#.apply(notifications),
    }
    source_data.update({names[i] : df[names[i]] for i in range(len(names))})
    source_data.update({'plot_' + names[i] : y_values[i] for i in range(len(names))})
    source_data2 = {'good_' + names[i] : df[circles[i]][df.notifications_mask == 0] for i in range(len(circles))}
    source_data2.update({'good_time' : df.row_timestamp[df.notifications_mask == 0]})
    source_data["skin_temp"] = df["skin_temp"]/2

    return source_data, source_data2, min_val, max_hr


def plot_data(df, dsn="AC000W002577443"):
    '''Plot the data in the given dataframe

    Parameters:
        df   (dataframe) - data from the given dsn
        dsn  (str) - device identifier

    '''
    y_values = [df.heart_rate_raw, df.oxygen_raw, df.skin_temp/2, df.base_state*10, df.movement_raw/2]
    names = ['heart_rate_raw', 'oxygen_raw', 'skin_temp', 'base_state', 'movement_raw']
    colors = ["blue", "orange", "red", "purple", "green"]
    alphas = [.25,.2,.25,.8,.5]
    circles = ["heart_rate_raw", "oxygen_raw"]

    source_data, source_data2, min_val, max_hr = get_source(df, y_values, names, circles)

    source = ColumnDataSource(data=source_data)
    source2 = ColumnDataSource(data=source_data2)

    hover_tool = HoverTool(
        tooltips=[
        ( 'time',   '@date{%T}' ),] +
        [(name, '@{}'.format(name)) for name in names] +
        [('notifications', '@notifications_mask')],

        formatters={'date'  : 'datetime',}, # use default 'numeral' formatter for other fields
        mode='vline', renderers=[]
    )

    # Vertical line where the mouse is
    crosshair = CrosshairTool(dimensions='height', line_alpha=.6)

    box_zoom = BoxZoomTool(dimensions='width')
    zoom_out = ZoomOutTool(dimensions='width', factor=.5)
    save = CustomSaveTool(save_name=dsn)

    # Create figure; use webgl for better rendering
    tools=[save, 'xpan', 'reset', hover_tool, crosshair, zoom_out, box_zoom]
    p = figure(width=950, height=500, title="{} Data".format(dsn), x_axis_type="datetime", tools=tools,
               toolbar_location="above", y_range=(max(0,min_val-10),max_hr+(24*(len(names)+1))), output_backend='webgl')

    # To have the Legend outside the plot, each line needs to be added to it
    legend_it = []
    cs = []
    for i in range(len(names)):
        legend_line = p.line(x=df.row_timestamp.iloc[-1:], y=y_values[i].iloc[-1:], color=colors[i], alpha=1,line_width=2)
        legend_it.append((names[i], [legend_line, p.line(x='date', y='plot_'+names[i], color=colors[i], alpha=alphas[i], source=source)]))
        if names[i] in circles:
            cs.append(p.circle(x='good_time', y='good_' + names[i], color=colors[i], size=1, alpha=0.7, source=source2))

    for i in range(len(circles)):
        legend_it.append(("valid_"+circles[i], [cs[i]]))


    # Creating a location for the tooltip box to appear (so it doesn't cover the data)
    horizontal_line = p.line(x='date', y='horizontal', color='white', alpha =0, source=source)
    hover_tool.renderers.append(horizontal_line)

    p.xaxis.axis_label = 'Time'
    p.xaxis.formatter=DatetimeTickFormatter(
            days=["%m/%d %T"], months=["%m/%d %T"],
            hours=["%m/%d %T"], minutes=["%m/%d %T"],
            seconds=["%m/%d %T"], minsec=["%m/%d %T"],
            hourmin=["%m/%d %T"]
    )

    legend = Legend(items=legend_it)
    legend.click_policy="hide"    # Hide lines when they are clicked on
    p.add_layout(legend, 'right')
    return p, source, source2

def update_data(p, text_title, text_save, source, source2, dsn, dates=None):
    names = ['heart_rate_raw', 'oxygen_raw', 'skin_temp', 'base_state', 'movement_raw']
    circles = ["heart_rate_raw", "oxygen_raw"]
    df = load_data(table="daily_digests", dsn_num=dsn, dates=dates)
    y_values = [df.heart_rate_raw, df.oxygen_raw, df.skin_temp/2, df.base_state*10, df.movement_raw/2]
    source.data, source2.data, min_val, max_hr = get_source(df, y_values, names, circles)

    # Title/save update dsn
    text_title.value = dsn + " Data"
    text_save.value = dsn

    # change y axis range:
    p.y_range.start = max(0,min_val-10)
    p.y_range.end = max_hr+(24*(len(names)+1))
    # return df? to change calendar to match dsn

def set_up_widgets(p, source, source2, dsn="AC000W002577443"):

    # Set up widgets
    text_title = TextInput(title="Title:", value="{} Data".format(dsn))
    text_save = TextInput(title="Save As:", value=dsn)
    text_dsn = TextInput(title="DSN:", value=dsn)

    # Initial values should be dynamic **********
    # get min and max row_timestamp, cut off time, split to get year, month, day
    calendar1 = DatePicker(title="Start Date:", value=date(2018, 1, 2), max_date=date(2018, 1, 5), min_date=date(2018, 1, 2))
    calendar2 = DatePicker(title="End Date:", value=date(2018, 1, 5), max_date=date(2018, 1, 5), min_date=date(2018, 1, 2))
    button = Button(label="Update", button_type="success")

    # Set up callbacks
    def update_title(attrname, old, new):
        p.title.text = text_title.value

    def update_save(attrname, old, new):
        p.tools[0].save_name = text_save.value

    def update():
        text_dsn.value = text_dsn.value.strip(" ")  # Get rid of extra space
        # Make sure time is valid
        date_start = "{} 00:00:00".format(calendar1.value)
        date_end = "{} 23:59:59".format(calendar2.value)

        update_data(p, text_title, text_save, source, source2, text_dsn.value, dates=[date_start, date_end])
        # Title/save update dsn
        text_title.value = text_dsn.value + " Data"
        text_save.value = text_dsn.value

    text_title.on_change('value', update_title)
    text_save.on_change('value', update_save)

    button.on_click(update)
    button.js_on_click(CustomJS(args=dict(p=p), code="""p.reset.emit()"""))

    # Set up layouts and add to document
    inputs = widgetbox(text_title, text_save, text_dsn, calendar1, calendar2, button)

    curdoc().add_root(row(inputs, p, width=1400))


df = load_data(table="daily_digests")
# df must not be empty
p, source, source2 = plot_data(df)#, y_values1, names1, colors1, alphas1, circles1)
# pass in df to widgets
set_up_widgets(p, source, source2)
