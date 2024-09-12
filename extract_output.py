import pandas as pd
import os

def shipment_logs(shipment_dict, actual_itinerary, actual_carried_shipments, assigned_to_rl, wait_actions, reassign_actions, late_data, episode, sd, policy_name, sim_nr):
    dfs = []

    # Iterate over each shipment in the dictionary
    for shipment in shipment_dict.values():
    # Calculate the total cost for the current shipment
        total_cost = (shipment.tot_shipment_storage_cost +
                    shipment.tot_shipment_handling_cost +
                    shipment.tot_shipment_travel_cost +
                    shipment.tot_shipment_delay_penalty)
        itinerary = actual_itinerary[shipment.name]
        if shipment.name in assigned_to_rl:
            rl = 'Yes'
        else:
            rl = 'No'
        missed_service = shipment.missed_service
        nr_wait = wait_actions[shipment.name]
        nr_reassign = reassign_actions[shipment.name]

        # Create a DataFrame for the current shipment
        df_shipment = pd.DataFrame({
            'Shipment': [shipment.name],
            'Storage Cost': [shipment.tot_shipment_storage_cost],
            'Handling Cost': [shipment.tot_shipment_handling_cost],
            'Travel Cost': [shipment.tot_shipment_travel_cost],
            'Delay Penalty': [shipment.tot_shipment_delay_penalty],
            'Total Cost': [total_cost],
            'Itinerary': [itinerary],
            'Assigned to RL': [rl],
            'Missed Service': [missed_service],
            'Wait Actions': [nr_wait],
            'Reassign Actions': [nr_reassign]
        })
        # Append the new DataFrame to the list
        dfs.append(df_shipment)

    # Concatenate all individual DataFrames into one
    df_shipment_costs = pd.concat(dfs, ignore_index=True)
    df_shipment_costs['Travel Cost'] = df_shipment_costs['Travel Cost'].round(2)
    df_shipment_costs['Handling Cost'] = df_shipment_costs['Handling Cost'].round(2)
    df_shipment_costs['Storage Cost'] = df_shipment_costs['Storage Cost'].round(2)
    df_shipment_costs['Delay Penalty'] = df_shipment_costs['Delay Penalty'].round(2)
    df_shipment_costs['Total Cost'] = df_shipment_costs['Total Cost'].round(2)

    # Crate service line dataframe
    df_service_line = pd.DataFrame(actual_carried_shipments.items(), columns=['Service Line', 'Number of Shipments'])
    late_data_df = pd.DataFrame.from_dict(late_data, orient='index', columns=['Late time', 'Number of late departure'])
    merged_df = df_service_line.merge(late_data_df, left_on='Service Line', right_index=True, how='left')

    folder_name = f'{sd}_{sim_nr}'
    folder_path_1 = 'shipment_logs'
    folder_path_2 = 'service_logs'
    if not os.path.exists(f'{folder_path_1}\\{folder_name}'):
        os.makedirs(f'{folder_path_1}\\{folder_name}')
    if not os.path.exists(f'{folder_path_2}\\{folder_name}'):
        os.makedirs(f'{folder_path_2}\\{folder_name}')

    # Export to csv
    df_shipment_costs.to_csv(f'{folder_path_1}\\{folder_name}\\shipment_output_{policy_name}_{episode}.csv', index=False)
    merged_df.to_csv(f'{folder_path_2}\\{folder_name}\\service_line_output_{policy_name}_{episode}.csv', index=False)