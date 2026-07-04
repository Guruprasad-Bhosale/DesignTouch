# TouchDesigner OSC In DAT Callback Script
# Assign this script to your OSC In DAT callback parameter.

def onReceive(dat, address, args, time, source):
    """
    Called whenever an OSC packet is received.
    Parses tracking and panel data and updates local TouchDesigner networks.
    """
    # Address format: /panels/<p_id>/corners, /panels/<p_id>/effect, /panels/<p_id>/alpha
    if address.startswith("/panels/"):
        parts = address.split("/")
        if len(parts) >= 4:
            panel_id = parts[2]  # e.g., 'band_1', 'spawned_1'
            param_type = parts[3]  # e.g., 'corners', 'effect', 'alpha'
            
            # Find or create a Table DAT named 'panel_data' to store states
            table = op("panel_data")
            if table is None:
                # If table doesn't exist, log warning
                print(f"[Warning] TouchDesigner requires a Table DAT named 'panel_data' to store panel coordinates.")
                return
                
            # Initialize table headers if empty
            if table.numRows == 0:
                table.appendRow(["id", "effect", "alpha", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
                
            # Find row index matching this panel_id
            row_idx = -1
            for r in range(1, table.numRows):
                if table[r, "id"] == panel_id:
                    row_idx = r
                    break
                    
            # If panel_id not found in table, append a new row
            if row_idx == -1:
                row_idx = table.numRows
                table.appendRow([panel_id, "ThermalVision", "0.0", "0", "0", "0", "0", "0", "0", "0", "0"])
                
            # Update specific parameters
            if param_type == "effect":
                table[row_idx, "effect"] = args[0]
            elif param_type == "alpha":
                table[row_idx, "alpha"] = args[0]
            elif param_type == "corners":
                # corners args contains 8 floats: [x0, y0, x1, y1, x2, y2, x3, y3]
                if len(args) >= 8:
                    for i in range(8):
                        col_name = f"{'x' if i%2==0 else 'y'}{i//2}"
                        table[row_idx, col_name] = args[i]
                        
    elif address == "/panels/count":
        # Can be used to sync active counts or cleanup expired panels
        count = args[0]
        table = op("panel_data")
        if table and table.numRows > 1:
            # We can purge spawned panels that are no longer sent
            # Wait for active panels list to be updated
            pass
            
    elif address == "/system/freeze":
        # Map freeze flag directly to switch TOP index
        freeze_val = args[0]
        switch_op = op("switch_freeze")
        if switch_op:
            switch_op.par.index = freeze_val
            
    return
