import json
import os
from typing import Dict
import geojson
import cv2
import numpy as np
import html

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def export_geojson(job: Dict, output_dir: str) -> str:
    """Generate GeoJSON FeatureCollection and validate it."""
    features = []
    
    eg = job.get("evidence_graph", {})
    if eg and "nodes" in eg:
        for node in eg["nodes"]:
            if node.get("type") == "evidence":
                region = node.get("attributes", {}).get("spatial_region")
                if region:
                    features.append(region)
                    
    feature_collection = geojson.FeatureCollection(features)
    
    # RFC 7946 validation check
    if not feature_collection.is_valid:
        print(f"Warning: GeoJSON generated for job {job['job_id']} is not fully RFC 7946 valid. Errors: {feature_collection.errors()}")
        
    filepath = os.path.join(output_dir, f"export_{job['job_id']}.geojson")
    with open(filepath, "w") as f:
        json.dump(feature_collection, f, indent=2)
        
    return filepath

def export_json_trace(job: Dict, output_dir: str) -> str:
    """Generate JSON execution trace."""
    filepath = os.path.join(output_dir, f"export_{job['job_id']}_trace.json")
    with open(filepath, "w") as f:
        json.dump({
            "job_id": job["job_id"],
            "status": job["status"],
            "trace": job["progress_trace"]
        }, f, indent=2)
    return filepath

def export_png_heatmap(job: Dict, output_dir: str) -> str:
    """Generate PNG heatmap overlay using OpenCV."""
    filepath = os.path.join(output_dir, f"export_{job['job_id']}_heatmap.png")
    
    # If we don't have a change_map in result, create a blank transparent PNG
    result = job.get("result", {})
    change_map = result.get("change_map")
    
    if not change_map:
        # Create empty 256x256 transparent PNG
        img = np.zeros((256, 256, 4), dtype=np.uint8)
        cv2.imwrite(filepath, img)
        return filepath
        
    # change_map is a list of lists of floats [0, 1]
    # Convert to numpy array
    change_array = np.array(change_map[0][0], dtype=np.float32) if len(np.shape(change_map)) == 4 else np.array(change_map, dtype=np.float32)
    
    # Squeeze out extra dims if present (e.g. if it was [1, 1, 256, 256])
    change_array = np.squeeze(change_array)
    
    # Scale to 0-255
    heatmap_gray = (change_array * 255).astype(np.uint8)
    
    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    
    # Create alpha channel (more transparent for lower values)
    alpha = heatmap_gray.copy()
    
    # Combine into BGRA
    b, g, r = cv2.split(heatmap_color)
    bgra = cv2.merge((b, g, r, alpha))
    
    cv2.imwrite(filepath, bgra)
    return filepath

def export_pdf_report(job: Dict, output_dir: str) -> str:
    """Generate PDF audit report using reportlab."""
    filepath = os.path.join(output_dir, f"export_{job['job_id']}_report.pdf")
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Title
    story.append(Paragraph(f"SatQuery AI - Analysis Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Job ID
    story.append(Paragraph(f"<b>Job ID:</b> {html.escape(str(job['job_id']))}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Final Result
    result = job.get("result", {})
    if result:
        story.append(Paragraph("<b>Final Answer:</b>", styles['Heading2']))
        ans = html.escape(str(result.get("final_answer", "No answer generated.")))
        story.append(Paragraph(ans, styles['Normal']))
        story.append(Spacer(1, 12))
        
        confidence = result.get("confidence", 0)
        story.append(Paragraph(f"<b>Confidence Score:</b> {confidence * 100:.1f}%", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Change Statistics
        stats = result.get("change_statistics")
        if stats:
            story.append(Paragraph("<b>Change Statistics:</b>", styles['Heading3']))
            story.append(Paragraph(f"- Changed Area: {stats.get('changed_area_m2', 0):.2f} m²", styles['Normal']))
            story.append(Paragraph(f"- Changed Pixels: {stats.get('changed_pixel_pct', 0) * 100:.2f}%", styles['Normal']))
            story.append(Spacer(1, 12))
        
        story.append(Paragraph("<b>Evidence Claims:</b>", styles['Heading3']))
        for claim in result.get("claims", []):
            story.append(Paragraph(f"- {html.escape(str(claim))}", styles['Normal']))
        story.append(Spacer(1, 12))
        
    else:
        story.append(Paragraph(f"<b>Status:</b> {html.escape(str(job['status']))}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Trace
    story.append(Paragraph("<b>Execution Trace Summary:</b>", styles['Heading2']))
    for entry in job.get("progress_trace", []):
        msg = html.escape(str(entry.get('message', '')))
        stage = html.escape(str(entry.get('stage', '')))
        story.append(Paragraph(f"[{stage}] {msg}", styles['Normal']))
        if entry.get("reason"):
            reason = html.escape(str(entry['reason']))
            story.append(Paragraph(f"<i>Reason: {reason}</i>", styles['Normal']))
            
    doc.build(story)
    return filepath

def generate_exports(job: Dict) -> Dict[str, str]:
    """Generates all requested export artifacts for a DONE/FAILED job."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "exports")
    os.makedirs(output_dir, exist_ok=True)
    
    exports = {}
    
    exports["json"] = export_json_trace(job, output_dir)
    
    if job.get("status") == "DONE":
        exports["geojson"] = export_geojson(job, output_dir)
        exports["pdf"] = export_pdf_report(job, output_dir)
        exports["png"] = export_png_heatmap(job, output_dir)
        
    return exports
