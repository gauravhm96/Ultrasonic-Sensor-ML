def general_tab(self):
    """Create the General tab."""
    tab = QWidget()
    layout = QVBoxLayout()

    label = QLabel("General Settings")
    label.setStyleSheet("font-size: 18px; font-weight: bold;")
    layout.addWidget(label)

    # Create a scrollable area for the features
    scroll_area = QScrollArea()  # Create the scrollable area
    scroll_area.setWidgetResizable(True)  # Allow content to resize

    # Create a container widget for the features to be added inside the scroll area
    feature_container = QWidget()
    feature_layout = QVBoxLayout()

    # Call the function from features.py to add more features to the layout
    add_general_features(feature_layout, self.output_box)  # This will add the additional features to the layout

    feature_container.setLayout(feature_layout)  # Set the layout of the container widget
    scroll_area.setWidget(feature_container)  # Set the feature container inside the scroll area

    layout.addWidget(scroll_area)  # Add scroll area to the main layout

    # Ensure the dialog box (output_box) does not stretch too much
    self.output_box.setFixedHeight(250)  # Make sure output_box height stays constant

    # Adjusting button layout (if needed)
    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(10, 10, 10, 10)
    button_layout.setSpacing(20)

    # Make buttons adjust more appropriately
    ok_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    exit_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # Add buttons to the layout
    button_layout.addWidget(ok_button)
    button_layout.addWidget(cancel_button)
    button_layout.addWidget(exit_button)

    layout.addLayout(button_layout)  # Add the button layout to the main layout

    layout.addStretch()  # Push contents to the top if needed
    tab.setLayout(layout)  # Set the layout to the tab widget

    return tab