import pandas as pd
from ipydatagrid import DataGrid, TextRenderer
from ipywidgets import Button, VBox, HBox, IntText, Output, Layout
from IPython.display import display

from projects.tutorial.settings import PROJECT_PATH

def get_table(source_file: str, sheet_name: str, target_file: str):
    # Load Excel file
    global df
    df = pd.read_excel(str(PROJECT_PATH) + source_file,
                       sheet_name=sheet_name)
    df.columns = df.iloc[0]
    df = df.loc[:, (df.iloc[4] != "no") & (df.iloc[4] != "no ")]

    # ToDo: skiprow is sheet dependent

    df_description = df.iloc[1:5]
    df_description = df_description.infer_objects(copy=False)
    df_description.loc[:, :] = df_description.fillna("")
    df_description.reset_index(drop=True, inplace=True)

    df = df.iloc[5:]
    df.iloc[:, 1:-1] = ""
    df.reset_index(drop=True, inplace=True)

    # Create DataGrid - for explanation
    grid_description = DataGrid(
        df_description,
        base_column_size=100,
        base_row_size=30,
        selection_mode='cell',
        editable=False,
    )

    # Create DataGrid - for state model
    grid = DataGrid(
        df,
        base_column_size=100,
        base_row_size=30,
        selection_mode='cell',
        editable=True,
    )

    # Input field for the row index
    row_index_input = IntText(
        value=0,
        description='Row index:',
        layout={'width': '200px'}
    )

    # Output for messages
    output = Output()

    def rebuild_grid():
        global grid
        # Remove old grid from main display
        main_display.children = main_display.children[:-1]
        # Create new grid
        grid = DataGrid(
            df,
            base_column_size=100,
            base_row_size=30,
            selection_mode='cell',
            editable=True,
        )
        # Add new grid to main display
        main_display.children += (grid,)

    def add_row_at_index(b):
        global df
        idx = int(row_index_input.value)
        print(idx)
        if idx < 0 or idx > len(df):
            with output:
                output.clear_output()
                print("Invalid index for adding.")
            return
        # Create new empty row
        new_row = pd.DataFrame([{col: "" for col in df.columns}])
        # Insert new row
        df = pd.concat([df.iloc[:idx], new_row, df.iloc[idx:]]).reset_index(drop=True)
        # Update DataGrid
        grid.data = df
        rebuild_grid()
        with output:
            output.clear_output()
            print(f"Row at index {idx} added.")

    def delete_row_at_index(b):
        global df
        idx = int(row_index_input.value)
        print(idx)
        if idx < 0 or idx >= len(df):
            with output:
                output.clear_output()
                print("Invalid index for deletion.")
            return
        # Delete row
        df = df.drop(df.index[idx]).reset_index(drop=True)
        # Update DataGrid
        grid.data = df
        rebuild_grid()
        with output:
            output.clear_output()
            print(f"Row at index {idx} deleted.")

    # Function to save changes
    def save_changes(b):
        edited_df = grid.data
        # Save changes to an Excel file
        edited_df.to_excel(str(PROJECT_PATH) + target_file, index=False)
        with output:
            output.clear_output()
            print("Changes have been saved.")

    # Buttons for adding and deleting
    add_row_button = Button(
        description='Add row',
        layout={'width': '150px'},
        button_style='success'  # Green button
        # Or use custom color:
        # style={'button_color': '#28a745'}
    )
    delete_row_button = Button(
        description='Delete row',
        layout={'width': '150px'},
        button_style='danger'  # Red button
        # Or use custom color:
        # style={'button_color': '#dc3545'}
    )
    save_button = Button(
        description='Save changes',
        layout={'width': '200px'},
        button_style='primary'  # Blue button
        # Or use custom color:
        # style={'button_color': '#007bff'}
    )

    # Link functions to buttons
    add_row_button.on_click(add_row_at_index)
    delete_row_button.on_click(delete_row_at_index)
    save_button.on_click(save_changes)

    # Display buttons and input field
    control_panel = HBox([row_index_input, add_row_button, delete_row_button, save_button])
    display(VBox([grid_description], layout=Layout(height="200px")))
    # Display everything together
    main_display = VBox([control_panel, output, grid])
    display(main_display)


def get_solution(source_file: str, sheet_name: str):
    solution_df = pd.read_excel(str(PROJECT_PATH) + source_file, sheet_name=sheet_name)
    solution_df.columns = solution_df.iloc[0]
    solution_df = solution_df.loc[:, (solution_df.iloc[4] != "no") & (solution_df.iloc[4] != "no ")]
    solution_df.loc[:, :] = solution_df.fillna("")
    # solution_df = solution_df.loc[:, ~solution_df.iloc[4].astype(str).str.contains('^no$|^$', na=False)]
    solution_df = solution_df.iloc[5:]
    solution_df.reset_index(drop=True, inplace=True)

    # Create DataGrid - for state model
    grid_solution = DataGrid(
        solution_df,
        base_column_size=150,
        base_row_size=30,
        selection_mode='cell',
        editable=False,
    )

    grid_solution.column_sizes = {
        'Description': 300,  # Example: Adjust the width of the 'Description' column
        'Name': 200,
        'Value': 250,
        # Add more columns as needed
    }

    # Create TextRenderer with line wrapping
    wrapped_text_renderer = TextRenderer(wrap_lines=True)

    # Assign the TextRenderer with line wrapping to specific columns
    grid_solution.cell_renderers = {
        ('Description', 'body'): wrapped_text_renderer,  # Example for the 'Description' column
        # Add more columns as needed
    }

    display(VBox([grid_solution], layout=Layout(height="200px")))

if __name__ == '__main__':
    get_table("/models/mini_model.xlsx",
              sheet_name="Sales",
              target_file="Sales_modeled.xlsx")

