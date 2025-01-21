def render_output(self):
        if self.state == "data_loaded":
            # User can opt to view the first few rows of the loaded data
            user_input = input("Do you want to view the first few rows of the signal data? (yes/no): ").strip().lower()
            if user_input == 'yes':
                print(self.signal_data.head())
            elif user_input != 'no':
                print("Invalid input. Please enter 'yes' or 'no'.")
        
        if self.state != "signals_analyzed":
            print("Plots can only be rendered in the 'signals_analyzed' state.")
            return

        # Ask the user for confirmation to display the plots
        user_input = input("Do you want to view the signal plots? (yes/no): ").strip().lower()

        if user_input == 'yes':
            try:
                # Create a single figure with 2 subplots
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))

                # Plot all the signals
                df = pd.DataFrame(self.signal_data)  # Ensure the signal data is in a DataFrame

                for i in range(df.shape[0]):
                    signal = df.iloc[i].iloc[16:]  # Ignoring the first 16 columns
                    axs[0].plot(signal.values[:])

                # Adding labels and title to the first subplot
                axs[0].grid(True)
                axs[0].set_xlabel('Time (samples)')
                axs[0].set_ylabel('Amplitude')
                axs[0].set_title('Signal Data')

                # Plot all the signals with absolute values and peaks
                for i in range(df.shape[0]):
                    signal = df.iloc[i].iloc[16:]  # Ignoring the first 16 columns
                    absolute_signal = signal.abs()  # Taking the absolute values

                    # Calculate mean and standard deviation of the absolute signal
                    mean_signal = np.mean(absolute_signal)
                    std_signal = np.std(absolute_signal)

                    # Set prominence threshold dynamically based on mean and standard deviation
                    prominence_threshold = mean_signal + 3 * std_signal  

                    peaks, _ = find_peaks(absolute_signal.values[:], prominence=prominence_threshold)

                    key = f'S{i+1}'
                    self.signal_dictionary.append((key, absolute_signal, peaks))

                    axs[1].plot(absolute_signal.values[:])
                    axs[1].plot(peaks, absolute_signal.values[:][peaks], 'x', color='green')  # Plotting the peaks for visualization

                # Adding labels and title to the second subplot
                axs[1].grid(True)
                axs[1].set_xlabel('Time (samples)')
                axs[1].set_ylabel('Absolute Amplitude')
                axs[1].set_title('Absolute Value of Signals With Peaks')

                # Add a main title to the entire figure
                fig.suptitle('Raw Data Captured from Ultrasonic Sensor', fontsize=16)

                # Adjust layout to prevent overlap and ensure proper layout
                plt.tight_layout()
                plt.show()

                self.state = "output_rendered"
                print("Plots rendered successfully.")
            except Exception as e:
                print(f"An error occurred while rendering the plots: {e}")
                self.state = "error"
        elif user_input == 'no':
            print("Render output skipped.")
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
