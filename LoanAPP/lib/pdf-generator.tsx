import html2canvas from "html2canvas"
import jsPDF from "jspdf"

export async function generatePDF(prediction: any) {
  try {
    // Create a temporary container with the report content
    const reportContent = document.createElement("div")
    reportContent.id = "pdf-report"
    // Reset inherited styles to avoid using modern color functions (e.g., oklch) from Tailwind v4
    // html2canvas currently cannot parse "oklch(...)" colors, so we fully isolate this node
    ;(reportContent.style as any).all = "initial"
    reportContent.style.display = "block"
    reportContent.style.padding = "20px"
    reportContent.style.backgroundColor = "white"
    reportContent.style.color = "#000000"
    reportContent.style.width = "800px"
    reportContent.style.position = "absolute"
    reportContent.style.left = "-9999px"
    reportContent.style.top = "0"
    reportContent.style.fontFamily = "Arial, sans-serif"
    const applicantName = prediction.applicant_name || "Applicant"
    const currentDate = new Date().toLocaleDateString()
    const riskLevel = prediction.default_probability > 0.7 ? "HIGH" : prediction.default_probability > 0.4 ? "MEDIUM" : "LOW"
    const statusColor = prediction.predicted_class === 1 ? "#dc2626" : "#16a34a"
    const statusText = prediction.predicted_class === 1 ? "DECLINED" : "APPROVED"
    
    reportContent.innerHTML = `
      <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 15px; background: white; font-size: 12px;">
        <!-- Header -->
        <div style="text-align: center; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 15px;">
          <h1 style="font-size: 18px; margin: 0; font-weight: bold;">LendSmart Bank</h1>
          <h2 style="font-size: 14px; margin: 5px 0 0 0; font-weight: normal;">Credit Risk Assessment Report</h2>
        </div>

        <!-- Patient/Applicant Info -->
        <div style="margin-bottom: 15px;">
          <table style="width: 100%; border-collapse: collapse;">
            <tr>
              <td style="padding: 3px 0; width: 30%;"><strong>Applicant:</strong></td>
              <td style="padding: 3px 0;">${applicantName}</td>
              <td style="padding: 3px 0; width: 30%;"><strong>Date:</strong></td>
              <td style="padding: 3px 0;">${currentDate}</td>
            </tr>
            <tr>
              <td style="padding: 3px 0;"><strong>Risk Level:</strong></td>
              <td style="padding: 3px 0; color: ${statusColor}; font-weight: bold;">${riskLevel}</td>
              <td style="padding: 3px 0;"><strong>Decision:</strong></td>
              <td style="padding: 3px 0; color: ${statusColor}; font-weight: bold;">${statusText}</td>
            </tr>
          </table>
        </div>

        <!-- Main Results -->
        <div style="margin-bottom: 15px;">
          <h3 style="font-size: 14px; margin: 0 0 8px 0; font-weight: bold;">Risk Assessment Results</h3>
          <table style="width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; border: 0.5px solid #999;">
            <tr style="background: #f5f5f5;">
              <th style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: left; font-size: 11px; width: 25%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Test</th>
              <th style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; width: 20%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Result</th>
              <th style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; width: 30%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Reference Range</th>
              <th style="border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; width: 25%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Status</th>
            </tr>
            <tr>
              <td style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; font-size: 11px; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Default Probability</td>
              <td style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; font-weight: bold; color: ${statusColor}; word-wrap: break-word; white-space: nowrap; overflow: hidden;">${(prediction.default_probability * 100).toFixed(1)}%</td>
              <td style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; word-wrap: break-word; white-space: nowrap; overflow: hidden;">0-40% (Low Risk)</td>
              <td style="border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; color: ${statusColor}; font-weight: bold; word-wrap: break-word; white-space: nowrap; overflow: hidden;">${prediction.default_probability < 0.4 ? 'NORMAL' : prediction.default_probability < 0.7 ? 'HIGH' : 'CRITICAL'}</td>
            </tr>
          </table>
        </div>

        <!-- Key Factors -->
        <div style="margin-bottom: 15px;">
          <h3 style="font-size: 14px; margin: 0 0 8px 0; font-weight: bold;">Key Risk Factors</h3>
          <table style="width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; border: 0.5px solid #999;">
            <tr style="background: #f5f5f5;">
              <th style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: left; font-size: 11px; width: 50%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Factor</th>
              <th style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; width: 25%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Impact</th>
              <th style="border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; width: 25%; word-wrap: break-word; white-space: nowrap; overflow: hidden;">Confidence</th>
        </tr>
        ${prediction.feature_contributions
              .slice(0, 10) // Show top 10 factors
          .map(
                (f: any, index: number) => `
              <tr>
                <td style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; font-size: 11px; word-wrap: break-word; white-space: nowrap; overflow: hidden;">${f.variable}</td>
                <td style="border-right: 0.5px solid #999; border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; color: ${f.contribution > 0 ? '#dc2626' : '#16a34a'}; word-wrap: break-word; white-space: nowrap; overflow: hidden;">
                  ${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)}
                </td>
                <td style="border-bottom: 0.5px solid #999; padding: 6px 4px; text-align: center; font-size: 11px; word-wrap: break-word; white-space: nowrap; overflow: hidden;">${(f.confidence * 100).toFixed(0)}%</td>
          </tr>
        `,
          )
          .join("")}
      </table>
        </div>

        <!-- Summary -->
        <div style="margin-bottom: 10px;">
          <h3 style="font-size: 14px; margin: 0 0 5px 0; font-weight: bold;">Summary</h3>
          <p style="margin: 0; font-size: 11px; line-height: 1.4;">${prediction.key_reason}</p>
        </div>

        ${prediction.verdict ? `
        <!-- AI Assessment -->
        <div style="margin-bottom: 10px;">
          <h3 style="font-size: 14px; margin: 0 0 5px 0; font-weight: bold;">AI Assessment</h3>
          <p style="margin: 0; font-size: 11px; line-height: 1.4; font-style: italic;">${prediction.verdict}</p>
        </div>
        ` : ""}

        <!-- Footer -->
        <div style="text-align: center; margin-top: 15px; padding-top: 8px; border-top: 1px solid #ccc; font-size: 10px; color: #666;">
          <p style="margin: 0;">LendSmart Bank AI Credit Analyzer | Generated: ${new Date().toLocaleString()}</p>
        </div>
      </div>
    `

    // Append to body temporarily
    document.body.appendChild(reportContent)

    // Convert HTML to canvas with better options
    const canvas = await html2canvas(reportContent, {
      backgroundColor: "#ffffff",
      scale: 2,
      useCORS: true,
      allowTaint: true,
      foreignObjectRendering: false,
      logging: false,
      width: 800,
      height: reportContent.scrollHeight,
      onclone: (clonedDoc: Document) => {
        try {
          // Remove global stylesheets (e.g., Tailwind v4) that define oklch colors
          const head = clonedDoc.querySelector("head")
          head?.querySelectorAll('link[rel="stylesheet"], style').forEach((el) => el.parentElement?.removeChild(el))

          // Force cloned body and container backgrounds to safe colors
          if (clonedDoc.body) {
            clonedDoc.body.style.backgroundColor = "#ffffff"
            clonedDoc.body.style.color = "#000000"
          }

          const clonedContainer = clonedDoc.getElementById("pdf-report") as HTMLElement | null
          if (clonedContainer) {
            clonedContainer.style.backgroundColor = "#ffffff"
            clonedContainer.style.color = "#000000"
          }
        } catch {}
      },
    })

    // Remove the temporary element
    document.body.removeChild(reportContent)

    // Create PDF
    const pdf = new jsPDF({
      orientation: "portrait",
      unit: "mm",
      format: "a4",
    })

    const imgData = canvas.toDataURL("image/png")
    const imgWidth = 210 // A4 width in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width
    let heightLeft = imgHeight

    let position = 0
    pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight)
    heightLeft -= 297 // A4 height in mm

    while (heightLeft >= 0) {
      position = heightLeft - imgHeight
      pdf.addPage()
      pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight)
      heightLeft -= 297
    }

    const safeFileName = `lendsmart-report-${applicantName}`.replace(/[^a-z0-9\-]+/gi, "_")
    pdf.save(`${safeFileName}.pdf`)
  } catch (error) {
    console.error("PDF generation failed:", error)
    // Clean up any remaining temporary elements
    const tempElements = document.querySelectorAll('[style*="position: absolute"][style*="left: -9999px"]')
    tempElements.forEach(el => el.remove())
    throw error
  }
}
