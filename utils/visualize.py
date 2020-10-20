import os
import numpy as np
import cv2

def visualize_results(img, z, boxes, masks, amodal_masks, zt, ze, cXs, cYs, diameters, diametersmm, idx, real_diameter):       
    masks = masks.astype(np.uint8)
    height, width = img.shape[:2]

    if masks.any():
        maskstransposed = masks.transpose(1,2,0)
        
        red_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        blue_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        green_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        all_masks = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],3),dtype=np.uint8)
        
        z_img_final = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.uint8)
        
        for i in range (maskstransposed.shape[-1]):
            masksel = np.expand_dims(maskstransposed[:,:,i],axis=2).astype(np.uint8)

            bbox = boxes[i].astype(np.uint16)
            masksel = masksel[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            zsel = z[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            z_mask = np.multiply(zsel,masksel)
            
            if zt.size > 0 and ze.size > 0:
                z_top = zt[i]
                z_edge = ze[i]
                z_mask_final = np.where(np.logical_and(z_mask>=z_top, z_mask<=z_edge),z_mask,0)
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) 

                z_img = z_mask_final.copy()
                color_range = 200
                if int(z_top) != 0 and int(z_edge) != 0:
                    np.clip(z_img, z_top, z_edge, out=z_img)
                    z_img = np.interp(z_img, (z_img.min(), z_img.max()), (0, 200))
                z_img = z_img.astype(np.uint8)
                z_img_final[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_img
            else:
                z_mask_final = z_mask
                z_mask_final_binary = np.minimum(z_mask_final,1).astype(np.uint8) 

            mask_diff = np.subtract(masksel,z_mask_final_binary)
            mask_diff = mask_diff.reshape(mask_diff.shape[0],mask_diff.shape[1])
            blue_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_diff

            z_mask_final_binary = z_mask_final_binary.reshape(z_mask_final_binary.shape[0],z_mask_final_binary.shape[1])
            green_mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = z_mask_final_binary

        all_masks[:,:,0] = blue_mask
        all_masks[:,:,1] = green_mask
        all_masks[:,:,2] = red_mask
        all_masks = np.multiply(all_masks,255).astype(np.uint8)

        z_img_final = np.where(np.absolute(z_img_final)==0, np.interp(z, (400, 1000), (50, 255)), z_img_final)
        z_img_final = np.multiply(z_img_final, np.minimum(z,1))
        z_img_final = z_img_final.astype(np.uint8)
        z3 = cv2.cvtColor(z_img_final,cv2.COLOR_GRAY2RGB)  

        img_mask = cv2.addWeighted(img,1,all_masks,0.5,0)
        zimg_mask = cv2.addWeighted(z3,1,all_masks,0.6,0)

        for k in range(cXs.size):
            if amodal_masks.size > 0:
                amodalmaskstransposed = amodal_masks.transpose(1,2,0)
                
                for j in range (amodalmaskstransposed.shape[-1]):
                    amodalmasksel = np.expand_dims(amodalmaskstransposed[:,:,j],axis=2).astype(np.uint8)
                    ret,broc_mask = cv2.threshold((amodalmasksel*255).astype(np.uint8),254,255,cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(broc_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_mask, contours, -1, (0, 0, 255), 3) 
                    cv2.drawContours(zimg_mask, contours, -1, (0, 0, 255), 3)

                    cnt = np.concatenate(contours)
                    M = cv2.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv2.circle(img_mask, (cX, cY), 7, (0, 0, 255), -1)
                    cv2.circle(zimg_mask, (cX, cY), 7, (0, 0, 255), -1)  
            else:
                cv2.circle(img_mask, (cXs[k], cYs[k]), 7, (0, 0, 255), -1)
                cv2.circle(img_mask, (cXs[k], cYs[k]), int(diameters[k]/2), (0, 0, 255), 3)

                cv2.circle(zimg_mask, (cXs[k], cYs[k]), 7, (0, 0, 255), -1)
                cv2.circle(zimg_mask, (cXs[k], cYs[k]), int(diameters[k]/2), (0, 0, 255), 3)
                

            bbox = boxes[k].astype(np.uint16)
            font_face = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            font_thickness = 1

            text_str2 = "Estimation: {:.1f} mm".format(diametersmm[k])
            text_w2, text_h2 = cv2.getTextSize(text_str2, font_face, font_scale, font_thickness)[0]

            if cXs[k] < (width/2):
                text_pt2 = (cXs[k] + int(diameters[k]/2) + 20, (cYs[k]+10))
            else:
                text_pt2 = (cXs[k] - int(diameters[k]/2) - 400, (cYs[k]+10))

            text_color1 = [255, 255, 255]
            text_color2 = [0, 0, 0]

            cv2.rectangle(img_mask, (text_pt2[0], text_pt2[1] + 7), (text_pt2[0] + text_w2, text_pt2[1] - text_h2 -7), text_color1, -1)
            cv2.putText(img_mask, text_str2, text_pt2, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

            cv2.rectangle(zimg_mask, (text_pt2[0], text_pt2[1] + 7), (text_pt2[0] + text_w2, text_pt2[1] - text_h2 - 7), text_color1, -1)
            cv2.putText(zimg_mask, text_str2, text_pt2, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

            if k == idx:
                text_str1 = "Real diameter: {} mm".format(real_diameter)
                text_str3 = "Difference: {:.1f} mm".format(np.subtract(float(real_diameter), diametersmm[k]))
                
                text_w1, text_h1 = cv2.getTextSize(text_str1, font_face, font_scale, font_thickness)[0]
                text_w3, text_h3 = cv2.getTextSize(text_str3, font_face, font_scale, font_thickness)[0]

                if cXs[k] < (width/2):
                    text_pt1 = (cXs[k] + int(diameters[k]/2) + 20, (cYs[k]-25))
                    text_pt3 = (cXs[k] + int(diameters[k]/2) + 20, (cYs[k]+45))
                else:
                    text_pt1 = (cXs[k] - int(diameters[k]/2) - 400, (cYs[k]-25))
                    text_pt3 = (cXs[k] - int(diameters[k]/2) - 400, (cYs[k]+45))

                cv2.rectangle(img_mask, (text_pt1[0], text_pt1[1] + 7), (text_pt1[0] + text_w1, text_pt1[1] - text_h1 - 7), text_color1, -1)
                cv2.putText(img_mask, text_str1, text_pt1, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

                cv2.rectangle(img_mask, (text_pt3[0], text_pt3[1] + 7), (text_pt3[0] + text_w3, text_pt3[1] - text_h3 -7), text_color1, -1)
                cv2.putText(img_mask, text_str3, text_pt3, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

                cv2.rectangle(zimg_mask, (text_pt1[0], text_pt1[1] + 7), (text_pt1[0] + text_w1, text_pt1[1] - text_h1 - 7), text_color1, -1)
                cv2.putText(zimg_mask, text_str1, text_pt1, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

                cv2.rectangle(zimg_mask, (text_pt3[0], text_pt3[1] + 7), (text_pt3[0] + text_w3, text_pt3[1] - text_h3 - 7), text_color1, -1)
                cv2.putText(zimg_mask, text_str3, text_pt3, font_face, font_scale, text_color2, font_thickness, cv2.LINE_AA)

    else:
        img_mask = img
        z_img_binary = np.minimum(z,1)
        zimg_mask = np.multiply(z_img_binary,200)
        zimg_mask = np.repeat(zimg_mask, 3, axis=2).astype(np.uint8)


    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    font_thickness = 2

    text_color1 = [255, 255, 255]
    text_pt4 = (int((width/2) - 200), 75)

    text_str4 = "RGB image"
    text_w4, text_h4 = cv2.getTextSize(text_str4, font_face, font_scale, font_thickness)[0]
    cv2.putText(img_mask, text_str4, text_pt4, font_face, font_scale, text_color1, font_thickness, cv2.LINE_AA)

    text_str5 = "Depth image"
    text_w5, text_h5 = cv2.getTextSize(text_str5, font_face, font_scale, font_thickness)[0]
    cv2.putText(zimg_mask, text_str5, text_pt4, font_face, font_scale, text_color1, font_thickness, cv2.LINE_AA)

    return img_mask, zimg_mask