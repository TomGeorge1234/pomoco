"""Generic plotting utilities for Matplotlib figures and axes."""

def outset_axes(ax, offset_mm=2):
    """
    Applies a constant physical offset (in mm) to the bottom (X) and 
    left (Y) axes spines of a Matplotlib Axes object, making them appear 
    'outset' from the data area.

    This achieves a constant absolute widthfor the offset, regardless of 
    data scale.

    Args:
        ax (matplotlib.axes.Axes): The axes object to modify.
        offset_mm (float): The distance in millimeters to offset
                          the axes from the plot area. Default is 4.
    """
    offset_pt = offset_mm * 72 / 25.4  # Convert mm to points

    # 1. Set Spines to 'outward' position
    # This pushes the spine line exactly 'offset_pt' distance outward.
    ax.spines['bottom'].set_position(('outward', offset_pt))
    ax.spines['left'].set_position(('outward', offset_pt))