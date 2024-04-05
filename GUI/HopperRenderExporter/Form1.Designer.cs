namespace HopperRenderExporter
{
    partial class mainWindow
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(mainWindow));
            openSourceVideoFileDialog = new OpenFileDialog();
            sourceVideoBrowseButton = new Button();
            sourceVideoFilePathTextBox = new TextBox();
            outputVideoFilePathTextBox = new TextBox();
            outputVideoBroseButton = new Button();
            outputVideoSaveFileDialog = new SaveFileDialog();
            inputLabel = new Label();
            outputLabel = new Label();
            hopperRenderExporterLabel = new Label();
            targetFPSlabel = new Label();
            targetFPSSelector = new NumericUpDown();
            speedLabel = new Label();
            speedSelector = new NumericUpDown();
            calcResLabel = new Label();
            calcResDivSelector = new NumericUpDown();
            numIterationsLabel = new Label();
            numIterationsSelector = new TrackBar();
            numStepsLabel = new Label();
            numStepsSelector = new TrackBar();
            frameBlurLabel = new Label();
            frameBlurKernelSelector = new TrackBar();
            flowBlurLabel = new Label();
            flowBlurKernelSelector = new TrackBar();
            frameOutputLabel = new Label();
            frameOutputSelector = new ComboBox();
            showPreviewCheckBox = new CheckBox();
            interpolateButton = new Button();
            process = new System.Diagnostics.Process();
            numSteps = new Label();
            frameBlur = new Label();
            flowBlur = new Label();
            numIterations = new Label();
            startTimeLabel = new Label();
            startTimeMinSelector = new NumericUpDown();
            startTimeMinLabel = new Label();
            startTimeSecSelector = new NumericUpDown();
            startTimeSecLabel = new Label();
            endTimeSecLabel = new Label();
            endTimeSecSelector = new NumericUpDown();
            endTimeMinLabel = new Label();
            endTimeMinSelector = new NumericUpDown();
            endTimeLabel = new Label();
            ((System.ComponentModel.ISupportInitialize)targetFPSSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)speedSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)calcResDivSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numIterationsSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)numStepsSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)frameBlurKernelSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)flowBlurKernelSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)startTimeMinSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)startTimeSecSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)endTimeSecSelector).BeginInit();
            ((System.ComponentModel.ISupportInitialize)endTimeMinSelector).BeginInit();
            SuspendLayout();
            // 
            // openSourceVideoFileDialog
            // 
            openSourceVideoFileDialog.ValidateNames = false;
            openSourceVideoFileDialog.FileOk += openFileDialog1_FileOk;
            // 
            // sourceVideoBrowseButton
            // 
            sourceVideoBrowseButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            sourceVideoBrowseButton.BackColor = Color.Gray;
            sourceVideoBrowseButton.FlatStyle = FlatStyle.Popup;
            sourceVideoBrowseButton.Font = new Font("Segoe UI", 12F);
            sourceVideoBrowseButton.ForeColor = Color.White;
            sourceVideoBrowseButton.Location = new Point(613, 64);
            sourceVideoBrowseButton.Name = "sourceVideoBrowseButton";
            sourceVideoBrowseButton.Size = new Size(75, 29);
            sourceVideoBrowseButton.TabIndex = 4;
            sourceVideoBrowseButton.Text = "Browse";
            sourceVideoBrowseButton.UseVisualStyleBackColor = false;
            sourceVideoBrowseButton.Click += sourceVideoBrowseButton_Click;
            // 
            // sourceVideoFilePathTextBox
            // 
            sourceVideoFilePathTextBox.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            sourceVideoFilePathTextBox.BackColor = Color.Gray;
            sourceVideoFilePathTextBox.BorderStyle = BorderStyle.FixedSingle;
            sourceVideoFilePathTextBox.Font = new Font("Segoe UI", 12F);
            sourceVideoFilePathTextBox.ForeColor = Color.White;
            sourceVideoFilePathTextBox.Location = new Point(69, 64);
            sourceVideoFilePathTextBox.Name = "sourceVideoFilePathTextBox";
            sourceVideoFilePathTextBox.Size = new Size(538, 29);
            sourceVideoFilePathTextBox.TabIndex = 1;
            sourceVideoFilePathTextBox.TextChanged += sourceVideoFilePathTextBox_TextChanged;
            // 
            // outputVideoFilePathTextBox
            // 
            outputVideoFilePathTextBox.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
            outputVideoFilePathTextBox.BackColor = Color.Gray;
            outputVideoFilePathTextBox.BorderStyle = BorderStyle.FixedSingle;
            outputVideoFilePathTextBox.Font = new Font("Segoe UI", 12F);
            outputVideoFilePathTextBox.ForeColor = Color.White;
            outputVideoFilePathTextBox.Location = new Point(69, 97);
            outputVideoFilePathTextBox.Name = "outputVideoFilePathTextBox";
            outputVideoFilePathTextBox.Size = new Size(538, 29);
            outputVideoFilePathTextBox.TabIndex = 2;
            // 
            // outputVideoBroseButton
            // 
            outputVideoBroseButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
            outputVideoBroseButton.BackColor = Color.Gray;
            outputVideoBroseButton.FlatStyle = FlatStyle.Popup;
            outputVideoBroseButton.Font = new Font("Segoe UI", 12F);
            outputVideoBroseButton.ForeColor = Color.White;
            outputVideoBroseButton.Location = new Point(613, 97);
            outputVideoBroseButton.Name = "outputVideoBroseButton";
            outputVideoBroseButton.Size = new Size(75, 29);
            outputVideoBroseButton.TabIndex = 3;
            outputVideoBroseButton.Text = "Browse";
            outputVideoBroseButton.UseVisualStyleBackColor = false;
            outputVideoBroseButton.Click += outputVideoBroseButton_Click;
            // 
            // inputLabel
            // 
            inputLabel.AutoSize = true;
            inputLabel.Font = new Font("Segoe UI", 12F);
            inputLabel.ForeColor = Color.White;
            inputLabel.Location = new Point(14, 67);
            inputLabel.Name = "inputLabel";
            inputLabel.Size = new Size(49, 21);
            inputLabel.TabIndex = 0;
            inputLabel.Text = "Input:";
            // 
            // outputLabel
            // 
            outputLabel.AutoSize = true;
            outputLabel.Font = new Font("Segoe UI", 12F);
            outputLabel.ForeColor = Color.White;
            outputLabel.Location = new Point(4, 100);
            outputLabel.Name = "outputLabel";
            outputLabel.Size = new Size(62, 21);
            outputLabel.TabIndex = 5;
            outputLabel.Text = "Output:";
            // 
            // hopperRenderExporterLabel
            // 
            hopperRenderExporterLabel.Anchor = AnchorStyles.Top;
            hopperRenderExporterLabel.AutoSize = true;
            hopperRenderExporterLabel.Font = new Font("Segoe UI", 21.75F, FontStyle.Bold, GraphicsUnit.Point, 0);
            hopperRenderExporterLabel.ForeColor = Color.White;
            hopperRenderExporterLabel.Location = new Point(172, 9);
            hopperRenderExporterLabel.Name = "hopperRenderExporterLabel";
            hopperRenderExporterLabel.Size = new Size(354, 40);
            hopperRenderExporterLabel.TabIndex = 6;
            hopperRenderExporterLabel.Text = "Hopper Render Exporter";
            hopperRenderExporterLabel.TextAlign = ContentAlignment.TopCenter;
            // 
            // targetFPSlabel
            // 
            targetFPSlabel.Anchor = AnchorStyles.Top;
            targetFPSlabel.AutoSize = true;
            targetFPSlabel.Font = new Font("Segoe UI", 12F);
            targetFPSlabel.ForeColor = Color.White;
            targetFPSlabel.Location = new Point(61, 145);
            targetFPSlabel.Name = "targetFPSlabel";
            targetFPSlabel.Size = new Size(82, 21);
            targetFPSlabel.TabIndex = 7;
            targetFPSlabel.Text = "Target FPS";
            // 
            // targetFPSSelector
            // 
            targetFPSSelector.Anchor = AnchorStyles.Top;
            targetFPSSelector.BackColor = Color.Gray;
            targetFPSSelector.DecimalPlaces = 3;
            targetFPSSelector.Font = new Font("Segoe UI", 12F);
            targetFPSSelector.ForeColor = Color.White;
            targetFPSSelector.Increment = new decimal(new int[] { 1, 0, 0, 196608 });
            targetFPSSelector.Location = new Point(44, 169);
            targetFPSSelector.Maximum = new decimal(new int[] { 2400, 0, 0, 65536 });
            targetFPSSelector.Minimum = new decimal(new int[] { 10, 0, 0, 65536 });
            targetFPSSelector.Name = "targetFPSSelector";
            targetFPSSelector.Size = new Size(120, 29);
            targetFPSSelector.TabIndex = 8;
            targetFPSSelector.Value = new decimal(new int[] { 600, 0, 0, 65536 });
            // 
            // speedLabel
            // 
            speedLabel.Anchor = AnchorStyles.Top;
            speedLabel.AutoSize = true;
            speedLabel.Font = new Font("Segoe UI", 12F);
            speedLabel.ForeColor = Color.White;
            speedLabel.Location = new Point(254, 145);
            speedLabel.Name = "speedLabel";
            speedLabel.Size = new Size(53, 21);
            speedLabel.TabIndex = 9;
            speedLabel.Text = "Speed";
            // 
            // speedSelector
            // 
            speedSelector.Anchor = AnchorStyles.Top;
            speedSelector.BackColor = Color.Gray;
            speedSelector.DecimalPlaces = 2;
            speedSelector.Font = new Font("Segoe UI", 12F);
            speedSelector.ForeColor = Color.White;
            speedSelector.Increment = new decimal(new int[] { 1, 0, 0, 131072 });
            speedSelector.Location = new Point(219, 169);
            speedSelector.Maximum = new decimal(new int[] { 40, 0, 0, 65536 });
            speedSelector.Minimum = new decimal(new int[] { 1, 0, 0, 131072 });
            speedSelector.Name = "speedSelector";
            speedSelector.Size = new Size(120, 29);
            speedSelector.TabIndex = 10;
            speedSelector.Value = new decimal(new int[] { 100, 0, 0, 131072 });
            // 
            // calcResLabel
            // 
            calcResLabel.Anchor = AnchorStyles.Top;
            calcResLabel.AutoSize = true;
            calcResLabel.Font = new Font("Segoe UI", 12F);
            calcResLabel.ForeColor = Color.White;
            calcResLabel.Location = new Point(390, 145);
            calcResLabel.Name = "calcResLabel";
            calcResLabel.Size = new Size(114, 21);
            calcResLabel.TabIndex = 11;
            calcResLabel.Text = "Calc Res Scalar";
            // 
            // calcResDivSelector
            // 
            calcResDivSelector.Anchor = AnchorStyles.Top;
            calcResDivSelector.BackColor = Color.Gray;
            calcResDivSelector.DecimalPlaces = 4;
            calcResDivSelector.Font = new Font("Segoe UI", 12F);
            calcResDivSelector.ForeColor = Color.White;
            calcResDivSelector.Increment = new decimal(new int[] { 1, 0, 0, 262144 });
            calcResDivSelector.Location = new Point(384, 169);
            calcResDivSelector.Maximum = new decimal(new int[] { 10, 0, 0, 65536 });
            calcResDivSelector.Minimum = new decimal(new int[] { 625, 0, 0, 262144 });
            calcResDivSelector.Name = "calcResDivSelector";
            calcResDivSelector.Size = new Size(120, 29);
            calcResDivSelector.TabIndex = 12;
            calcResDivSelector.Value = new decimal(new int[] { 10, 0, 0, 65536 });
            calcResDivSelector.ValueChanged += calcResDivSelector_ValueChanged;
            // 
            // numIterationsLabel
            // 
            numIterationsLabel.Anchor = AnchorStyles.Top;
            numIterationsLabel.AutoSize = true;
            numIterationsLabel.Font = new Font("Segoe UI", 12F);
            numIterationsLabel.ForeColor = Color.White;
            numIterationsLabel.Location = new Point(559, 208);
            numIterationsLabel.Name = "numIterationsLabel";
            numIterationsLabel.Size = new Size(110, 21);
            numIterationsLabel.TabIndex = 13;
            numIterationsLabel.Text = "NumIterations";
            // 
            // numIterationsSelector
            // 
            numIterationsSelector.Anchor = AnchorStyles.Top;
            numIterationsSelector.LargeChange = 2;
            numIterationsSelector.Location = new Point(559, 234);
            numIterationsSelector.Maximum = 14;
            numIterationsSelector.Name = "numIterationsSelector";
            numIterationsSelector.Size = new Size(104, 45);
            numIterationsSelector.TabIndex = 14;
            numIterationsSelector.Scroll += numIterationsSelector_Scroll;
            // 
            // numStepsLabel
            // 
            numStepsLabel.Anchor = AnchorStyles.Top;
            numStepsLabel.AutoSize = true;
            numStepsLabel.Font = new Font("Segoe UI", 12F);
            numStepsLabel.ForeColor = Color.White;
            numStepsLabel.Location = new Point(61, 208);
            numStepsLabel.Name = "numStepsLabel";
            numStepsLabel.Size = new Size(82, 21);
            numStepsLabel.TabIndex = 15;
            numStepsLabel.Text = "NumSteps";
            // 
            // numStepsSelector
            // 
            numStepsSelector.Anchor = AnchorStyles.Top;
            numStepsSelector.LargeChange = 1;
            numStepsSelector.Location = new Point(45, 234);
            numStepsSelector.Maximum = 15;
            numStepsSelector.Minimum = 4;
            numStepsSelector.Name = "numStepsSelector";
            numStepsSelector.Size = new Size(104, 45);
            numStepsSelector.TabIndex = 16;
            numStepsSelector.Value = 15;
            numStepsSelector.Scroll += numStepsSelector_Scroll;
            // 
            // frameBlurLabel
            // 
            frameBlurLabel.Anchor = AnchorStyles.Top;
            frameBlurLabel.AutoSize = true;
            frameBlurLabel.Font = new Font("Segoe UI", 12F);
            frameBlurLabel.ForeColor = Color.White;
            frameBlurLabel.Location = new Point(219, 208);
            frameBlurLabel.Name = "frameBlurLabel";
            frameBlurLabel.Size = new Size(126, 21);
            frameBlurLabel.TabIndex = 17;
            frameBlurLabel.Text = "FrameBlurKernel";
            // 
            // frameBlurKernelSelector
            // 
            frameBlurKernelSelector.Anchor = AnchorStyles.Top;
            frameBlurKernelSelector.LargeChange = 4;
            frameBlurKernelSelector.Location = new Point(227, 234);
            frameBlurKernelSelector.Maximum = 32;
            frameBlurKernelSelector.Name = "frameBlurKernelSelector";
            frameBlurKernelSelector.Size = new Size(104, 45);
            frameBlurKernelSelector.TabIndex = 18;
            frameBlurKernelSelector.Value = 16;
            frameBlurKernelSelector.Scroll += frameBlurKernelSelector_Scroll;
            // 
            // flowBlurLabel
            // 
            flowBlurLabel.Anchor = AnchorStyles.Top;
            flowBlurLabel.AutoSize = true;
            flowBlurLabel.Font = new Font("Segoe UI", 12F);
            flowBlurLabel.ForeColor = Color.White;
            flowBlurLabel.Location = new Point(386, 208);
            flowBlurLabel.Name = "flowBlurLabel";
            flowBlurLabel.Size = new Size(115, 21);
            flowBlurLabel.TabIndex = 19;
            flowBlurLabel.Text = "FlowBlurKernel";
            // 
            // flowBlurKernelSelector
            // 
            flowBlurKernelSelector.Anchor = AnchorStyles.Top;
            flowBlurKernelSelector.LargeChange = 4;
            flowBlurKernelSelector.Location = new Point(390, 234);
            flowBlurKernelSelector.Maximum = 32;
            flowBlurKernelSelector.Name = "flowBlurKernelSelector";
            flowBlurKernelSelector.Size = new Size(104, 45);
            flowBlurKernelSelector.TabIndex = 20;
            flowBlurKernelSelector.Value = 32;
            flowBlurKernelSelector.Scroll += flowBlurKernelSelector_Scroll;
            // 
            // frameOutputLabel
            // 
            frameOutputLabel.Anchor = AnchorStyles.Top;
            frameOutputLabel.AutoSize = true;
            frameOutputLabel.Font = new Font("Segoe UI", 12F);
            frameOutputLabel.ForeColor = Color.White;
            frameOutputLabel.Location = new Point(562, 145);
            frameOutputLabel.Name = "frameOutputLabel";
            frameOutputLabel.Size = new Size(107, 21);
            frameOutputLabel.TabIndex = 21;
            frameOutputLabel.Text = "Frame Output";
            // 
            // frameOutputSelector
            // 
            frameOutputSelector.Anchor = AnchorStyles.Top;
            frameOutputSelector.BackColor = Color.Gray;
            frameOutputSelector.FlatStyle = FlatStyle.Flat;
            frameOutputSelector.Font = new Font("Segoe UI", 12F);
            frameOutputSelector.ForeColor = Color.White;
            frameOutputSelector.FormattingEnabled = true;
            frameOutputSelector.Items.AddRange(new object[] { "Warped Frame 1->2", "Warped Frame 2->1", "Blended Frame", "HSV Flow", "Blurred Frame", "Side-by-Side 1", "Side-by-Side 2" });
            frameOutputSelector.Location = new Point(555, 169);
            frameOutputSelector.Name = "frameOutputSelector";
            frameOutputSelector.Size = new Size(121, 29);
            frameOutputSelector.TabIndex = 22;
            // 
            // showPreviewCheckBox
            // 
            showPreviewCheckBox.Anchor = AnchorStyles.Top;
            showPreviewCheckBox.AutoSize = true;
            showPreviewCheckBox.Font = new Font("Segoe UI", 12F);
            showPreviewCheckBox.ForeColor = Color.White;
            showPreviewCheckBox.Location = new Point(297, 380);
            showPreviewCheckBox.Name = "showPreviewCheckBox";
            showPreviewCheckBox.Size = new Size(127, 25);
            showPreviewCheckBox.TabIndex = 23;
            showPreviewCheckBox.Text = "Show Preview";
            showPreviewCheckBox.UseVisualStyleBackColor = true;
            // 
            // interpolateButton
            // 
            interpolateButton.Anchor = AnchorStyles.Top;
            interpolateButton.BackColor = Color.Green;
            interpolateButton.FlatStyle = FlatStyle.Popup;
            interpolateButton.Font = new Font("Segoe UI", 15.75F, FontStyle.Bold, GraphicsUnit.Point, 0);
            interpolateButton.ForeColor = Color.White;
            interpolateButton.Location = new Point(254, 407);
            interpolateButton.Name = "interpolateButton";
            interpolateButton.Size = new Size(191, 49);
            interpolateButton.TabIndex = 24;
            interpolateButton.Text = "Interpolate";
            interpolateButton.UseVisualStyleBackColor = false;
            interpolateButton.Click += interpolateButton_Click;
            // 
            // process
            // 
            process.StartInfo.Domain = "";
            process.StartInfo.LoadUserProfile = false;
            process.StartInfo.Password = null;
            process.StartInfo.StandardErrorEncoding = null;
            process.StartInfo.StandardInputEncoding = null;
            process.StartInfo.StandardOutputEncoding = null;
            process.StartInfo.UseCredentialsForNetworkingOnly = false;
            process.StartInfo.UserName = "";
            process.SynchronizingObject = this;
            // 
            // numSteps
            // 
            numSteps.Anchor = AnchorStyles.Top;
            numSteps.AutoSize = true;
            numSteps.Font = new Font("Segoe UI", 12F);
            numSteps.ForeColor = Color.White;
            numSteps.Location = new Point(80, 263);
            numSteps.Name = "numSteps";
            numSteps.Size = new Size(28, 21);
            numSteps.TabIndex = 26;
            numSteps.Text = "15";
            // 
            // frameBlur
            // 
            frameBlur.Anchor = AnchorStyles.Top;
            frameBlur.AutoSize = true;
            frameBlur.Font = new Font("Segoe UI", 12F);
            frameBlur.ForeColor = Color.White;
            frameBlur.Location = new Point(265, 263);
            frameBlur.Name = "frameBlur";
            frameBlur.Size = new Size(28, 21);
            frameBlur.TabIndex = 27;
            frameBlur.Text = "16";
            // 
            // flowBlur
            // 
            flowBlur.Anchor = AnchorStyles.Top;
            flowBlur.AutoSize = true;
            flowBlur.Font = new Font("Segoe UI", 12F);
            flowBlur.ForeColor = Color.White;
            flowBlur.Location = new Point(426, 263);
            flowBlur.Name = "flowBlur";
            flowBlur.Size = new Size(28, 21);
            flowBlur.TabIndex = 28;
            flowBlur.Text = "32";
            // 
            // numIterations
            // 
            numIterations.Anchor = AnchorStyles.Top;
            numIterations.AutoSize = true;
            numIterations.Font = new Font("Segoe UI", 12F);
            numIterations.ForeColor = Color.White;
            numIterations.Location = new Point(593, 263);
            numIterations.Name = "numIterations";
            numIterations.Size = new Size(35, 21);
            numIterations.TabIndex = 29;
            numIterations.Text = "Full";
            // 
            // startTimeLabel
            // 
            startTimeLabel.Anchor = AnchorStyles.Top;
            startTimeLabel.AutoSize = true;
            startTimeLabel.Font = new Font("Segoe UI", 12F);
            startTimeLabel.ForeColor = Color.White;
            startTimeLabel.Location = new Point(176, 292);
            startTimeLabel.Name = "startTimeLabel";
            startTimeLabel.Size = new Size(80, 21);
            startTimeLabel.TabIndex = 30;
            startTimeLabel.Text = "Start Time";
            // 
            // startTimeMinSelector
            // 
            startTimeMinSelector.Anchor = AnchorStyles.Top;
            startTimeMinSelector.BackColor = Color.Gray;
            startTimeMinSelector.Font = new Font("Segoe UI", 12F);
            startTimeMinSelector.ForeColor = Color.White;
            startTimeMinSelector.Location = new Point(150, 320);
            startTimeMinSelector.Name = "startTimeMinSelector";
            startTimeMinSelector.Size = new Size(52, 29);
            startTimeMinSelector.TabIndex = 31;
            // 
            // startTimeMinLabel
            // 
            startTimeMinLabel.Anchor = AnchorStyles.Top;
            startTimeMinLabel.AutoSize = true;
            startTimeMinLabel.Font = new Font("Segoe UI", 12F);
            startTimeMinLabel.ForeColor = Color.White;
            startTimeMinLabel.Location = new Point(201, 322);
            startTimeMinLabel.Name = "startTimeMinLabel";
            startTimeMinLabel.Size = new Size(37, 21);
            startTimeMinLabel.TabIndex = 32;
            startTimeMinLabel.Text = "min";
            // 
            // startTimeSecSelector
            // 
            startTimeSecSelector.Anchor = AnchorStyles.Top;
            startTimeSecSelector.BackColor = Color.Gray;
            startTimeSecSelector.Font = new Font("Segoe UI", 12F);
            startTimeSecSelector.ForeColor = Color.White;
            startTimeSecSelector.Location = new Point(233, 320);
            startTimeSecSelector.Maximum = new decimal(new int[] { 59, 0, 0, 0 });
            startTimeSecSelector.Name = "startTimeSecSelector";
            startTimeSecSelector.Size = new Size(52, 29);
            startTimeSecSelector.TabIndex = 33;
            // 
            // startTimeSecLabel
            // 
            startTimeSecLabel.Anchor = AnchorStyles.Top;
            startTimeSecLabel.AutoSize = true;
            startTimeSecLabel.Font = new Font("Segoe UI", 12F);
            startTimeSecLabel.ForeColor = Color.White;
            startTimeSecLabel.Location = new Point(286, 322);
            startTimeSecLabel.Name = "startTimeSecLabel";
            startTimeSecLabel.Size = new Size(32, 21);
            startTimeSecLabel.TabIndex = 34;
            startTimeSecLabel.Text = "sec";
            // 
            // endTimeSecLabel
            // 
            endTimeSecLabel.Anchor = AnchorStyles.Top;
            endTimeSecLabel.AutoSize = true;
            endTimeSecLabel.Font = new Font("Segoe UI", 12F);
            endTimeSecLabel.ForeColor = Color.White;
            endTimeSecLabel.Location = new Point(516, 322);
            endTimeSecLabel.Name = "endTimeSecLabel";
            endTimeSecLabel.Size = new Size(32, 21);
            endTimeSecLabel.TabIndex = 39;
            endTimeSecLabel.Text = "sec";
            // 
            // endTimeSecSelector
            // 
            endTimeSecSelector.Anchor = AnchorStyles.Top;
            endTimeSecSelector.BackColor = Color.Gray;
            endTimeSecSelector.Font = new Font("Segoe UI", 12F);
            endTimeSecSelector.ForeColor = Color.White;
            endTimeSecSelector.Location = new Point(463, 320);
            endTimeSecSelector.Maximum = new decimal(new int[] { 59, 0, 0, 0 });
            endTimeSecSelector.Name = "endTimeSecSelector";
            endTimeSecSelector.Size = new Size(52, 29);
            endTimeSecSelector.TabIndex = 38;
            // 
            // endTimeMinLabel
            // 
            endTimeMinLabel.Anchor = AnchorStyles.Top;
            endTimeMinLabel.AutoSize = true;
            endTimeMinLabel.Font = new Font("Segoe UI", 12F);
            endTimeMinLabel.ForeColor = Color.White;
            endTimeMinLabel.Location = new Point(430, 322);
            endTimeMinLabel.Name = "endTimeMinLabel";
            endTimeMinLabel.Size = new Size(37, 21);
            endTimeMinLabel.TabIndex = 37;
            endTimeMinLabel.Text = "min";
            // 
            // endTimeMinSelector
            // 
            endTimeMinSelector.Anchor = AnchorStyles.Top;
            endTimeMinSelector.BackColor = Color.Gray;
            endTimeMinSelector.Font = new Font("Segoe UI", 12F);
            endTimeMinSelector.ForeColor = Color.White;
            endTimeMinSelector.Location = new Point(379, 320);
            endTimeMinSelector.Name = "endTimeMinSelector";
            endTimeMinSelector.Size = new Size(52, 29);
            endTimeMinSelector.TabIndex = 36;
            // 
            // endTimeLabel
            // 
            endTimeLabel.Anchor = AnchorStyles.Top;
            endTimeLabel.AutoSize = true;
            endTimeLabel.Font = new Font("Segoe UI", 12F);
            endTimeLabel.ForeColor = Color.White;
            endTimeLabel.Location = new Point(421, 292);
            endTimeLabel.Name = "endTimeLabel";
            endTimeLabel.Size = new Size(74, 21);
            endTimeLabel.TabIndex = 35;
            endTimeLabel.Text = "End Time";
            // 
            // mainWindow
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = Color.FromArgb(64, 64, 64);
            ClientSize = new Size(698, 486);
            Controls.Add(endTimeSecLabel);
            Controls.Add(endTimeSecSelector);
            Controls.Add(endTimeMinLabel);
            Controls.Add(endTimeMinSelector);
            Controls.Add(endTimeLabel);
            Controls.Add(startTimeSecLabel);
            Controls.Add(startTimeSecSelector);
            Controls.Add(startTimeMinLabel);
            Controls.Add(startTimeMinSelector);
            Controls.Add(startTimeLabel);
            Controls.Add(numIterations);
            Controls.Add(flowBlur);
            Controls.Add(frameBlur);
            Controls.Add(numSteps);
            Controls.Add(interpolateButton);
            Controls.Add(showPreviewCheckBox);
            Controls.Add(frameOutputSelector);
            Controls.Add(frameOutputLabel);
            Controls.Add(flowBlurKernelSelector);
            Controls.Add(flowBlurLabel);
            Controls.Add(frameBlurKernelSelector);
            Controls.Add(frameBlurLabel);
            Controls.Add(numStepsSelector);
            Controls.Add(numStepsLabel);
            Controls.Add(numIterationsSelector);
            Controls.Add(numIterationsLabel);
            Controls.Add(calcResDivSelector);
            Controls.Add(calcResLabel);
            Controls.Add(speedSelector);
            Controls.Add(speedLabel);
            Controls.Add(targetFPSSelector);
            Controls.Add(targetFPSlabel);
            Controls.Add(hopperRenderExporterLabel);
            Controls.Add(outputLabel);
            Controls.Add(inputLabel);
            Controls.Add(outputVideoBroseButton);
            Controls.Add(outputVideoFilePathTextBox);
            Controls.Add(sourceVideoFilePathTextBox);
            Controls.Add(sourceVideoBrowseButton);
            Icon = (Icon)resources.GetObject("$this.Icon");
            MinimumSize = new Size(714, 525);
            Name = "mainWindow";
            Text = "Hopper Render Exporter";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)targetFPSSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)speedSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)calcResDivSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)numIterationsSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)numStepsSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)frameBlurKernelSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)flowBlurKernelSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)startTimeMinSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)startTimeSecSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)endTimeSecSelector).EndInit();
            ((System.ComponentModel.ISupportInitialize)endTimeMinSelector).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private OpenFileDialog openSourceVideoFileDialog;
        private Button sourceVideoBrowseButton;
        private TextBox sourceVideoFilePathTextBox;
        private TextBox outputVideoFilePathTextBox;
        private Button outputVideoBroseButton;
        private SaveFileDialog outputVideoSaveFileDialog;
        private Label inputLabel;
        private Label outputLabel;
        private Label hopperRenderExporterLabel;
        private Label targetFPSlabel;
        private NumericUpDown targetFPSSelector;
        private Label speedLabel;
        private NumericUpDown speedSelector;
        private Label calcResLabel;
        private NumericUpDown calcResDivSelector;
        private Label numIterationsLabel;
        private TrackBar numIterationsSelector;
        private Label numStepsLabel;
        private TrackBar numStepsSelector;
        private Label frameBlurLabel;
        private TrackBar frameBlurKernelSelector;
        private Label flowBlurLabel;
        private TrackBar flowBlurKernelSelector;
        private Label frameOutputLabel;
        private ComboBox frameOutputSelector;
        private CheckBox showPreviewCheckBox;
        private Button interpolateButton;
        private System.Diagnostics.Process process;
        private Label frameBlur;
        private Label numSteps;
        private Label numIterations;
        private Label flowBlur;
        private NumericUpDown startTimeMinSelector;
        private Label startTimeLabel;
        private Label startTimeMinLabel;
        private Label endTimeSecLabel;
        private NumericUpDown endTimeSecSelector;
        private Label endTimeMinLabel;
        private NumericUpDown endTimeMinSelector;
        private Label endTimeLabel;
        private Label startTimeSecLabel;
        private NumericUpDown startTimeSecSelector;
    }
}
