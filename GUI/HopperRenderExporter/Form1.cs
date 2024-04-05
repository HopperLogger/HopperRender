using System.ComponentModel;
using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace HopperRenderExporter
{
    public partial class mainWindow : Form
    {
        string exePath;
        public mainWindow()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // Check if the executable exists
            if (System.IO.File.Exists("..\\..\\..\\..\\..\\x64\\Exporter\\Exporter.exe"))
            {
                exePath = "..\\..\\..\\..\\..\\x64\\Exporter\\Exporter.exe";
            } else if (System.IO.File.Exists("Exporter.exe")) {
                exePath = "Exporter.exe";
            } else {
                MessageBox.Show("The Exporter executable was not found. Please make sure the executable is in the correct location.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                this.Close();
            }
        }

        private void openFileDialog1_FileOk(object sender, System.ComponentModel.CancelEventArgs e)
        {

        }

        private void sourceVideoBrowseButton_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog1 = new OpenFileDialog();
            openFileDialog1.Filter = "Video Files (*.mkv, *.mp4, *.ts)|*.mkv;*.mp4;*.ts|All files (*.*)|*.*";
            openFileDialog1.Title = "Select a Video File";
            openFileDialog1.ShowDialog();
            sourceVideoFilePathTextBox.Text = openFileDialog1.FileName;
        }

        private void sourceVideoFilePathTextBox_TextChanged(object sender, EventArgs e)
        {

        }

        private void outputVideoBroseButton_Click(object sender, EventArgs e)
        {
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();
            saveFileDialog1.Filter = "Video Files (*.mkv, *.mp4, *.ts)|*.mkv;*.mp4;*.ts|All files (*.*)|*.*";
            saveFileDialog1.Title = "Save the Rendered Video File";
            saveFileDialog1.ShowDialog();
            outputVideoFilePathTextBox.Text = saveFileDialog1.FileName;
        }

        private async void interpolateButton_Click(object sender, EventArgs e)
        {
            if (sourceVideoFilePathTextBox.Text == "")
            {
                MessageBox.Show("Please select a source video file.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            if (outputVideoFilePathTextBox.Text == "")
            {
                MessageBox.Show("Please select an output video file.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            if (frameOutputSelector.SelectedIndex == -1)
            {
                MessageBox.Show("Please select a frame output option.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            interpolateButton.Enabled = false;

            String sourceVideoFilePath = sourceVideoFilePathTextBox.Text;
            String outputVideoFilePath = outputVideoFilePathTextBox.Text;
            double targetFPS = Convert.ToDouble(targetFPSSelector.Value);
            double speed = Convert.ToDouble(speedSelector.Value);
            float calcResDiv = Convert.ToSingle(calcResDivSelector.Value);
            int numIterations = Convert.ToInt32(numIterationsSelector.Value);
            int numSteps = Convert.ToInt32(numStepsSelector.Value);
            int frameBlurKernel = Convert.ToInt32(frameBlurKernelSelector.Value);
            int flowBlurKernel = Convert.ToInt32(flowBlurKernelSelector.Value);
            int frameOutput = Convert.ToInt32(frameOutputSelector.SelectedIndex);
            int startTimeMin = Convert.ToInt32(startTimeMinSelector.Value);
            int startTimeSec = Convert.ToInt32(startTimeSecSelector.Value);
            int endTimeMin = Convert.ToInt32(endTimeMinSelector.Value);
            int endTimeSec = Convert.ToInt32(endTimeSecSelector.Value);
            bool showPreview = showPreviewCheckBox.Checked;

            // Command line arguments to pass to the executable
            string arguments = "\"" + sourceVideoFilePath + "\" \"" + outputVideoFilePath + "\" " + targetFPS + " " + speed + " " + calcResDiv + " " + numIterations + " " + numSteps + " " + frameBlurKernel + " " + flowBlurKernel + " " + frameOutput + " " + startTimeMin + " " + startTimeSec + " " + endTimeMin + " " + endTimeSec + " " + showPreview;

            // Start the process
            process.StartInfo.FileName = exePath;
            process.StartInfo.Arguments = arguments;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.RedirectStandardOutput = false;
            process.StartInfo.RedirectStandardError = false;

            process.Start();
            await Task.Run(() => process.WaitForExit());

            interpolateButton.Enabled = true;
        }

        private void numStepsSelector_Scroll(object sender, EventArgs e)
        {
            numSteps.Text = numStepsSelector.Value.ToString();
        }

        private void frameBlurKernelSelector_Scroll(object sender, EventArgs e)
        {
            frameBlur.Text = frameBlurKernelSelector.Value.ToString();
        }

        private void flowBlurKernelSelector_Scroll(object sender, EventArgs e)
        {
            flowBlur.Text = flowBlurKernelSelector.Value.ToString();
        }

        private void numIterationsSelector_Scroll(object sender, EventArgs e)
        {
            if (numIterationsSelector.Value == 0)
            {
                numIterations.Text = "Full";
            }
            else
            {
                numIterations.Text = numIterationsSelector.Value.ToString();
            }

        }

        private void calcResDivSelector_ValueChanged(object sender, EventArgs e)
        {

        }
    }
}