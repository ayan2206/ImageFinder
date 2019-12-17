//
//  ViewController.swift
//  ImageUpload
//
//  Created by Ayan Mandal on 12/17/19.
//  Copyright © 2019 1stdibs. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet private weak var myLabel: UILabel!
    @IBOutlet private weak var myButton: UIButton!
    @IBOutlet private weak var imageView: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    @IBAction private func uploadImage(_ sender: UIButton) {
        startUploading()
    }
    
    @IBAction private func reset(_ sender: UIButton) {
        self.myLabel.text = "Upload Now"
    }
    
    private lazy var sharedSession: URLSession = {
        let configuration = URLSessionConfiguration.default
        let urlSession = URLSession(configuration: configuration)
        return urlSession
    }()
    
    private func startUploading() {
        guard let uploadUrl = URL(string: "http://127.0.0.1:5000/upload-image") else {
            print("Invalid URL")
            return
        }
        
        let request = NSMutableURLRequest(url:uploadUrl)
        request.httpMethod = "POST"

        let boundary = generateBoundaryString()

        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        guard let imageData = imageView.image?.jpegData(compressionQuality: 1) else {
            print("Invalid ImageData")
            return
        }


        request.httpBody = createBodyWithParameters(parameters: nil, filePathKey: "file", imageDataKey: imageData as NSData, boundary: boundary) as Data
        
        let dataToUpload = createBodyWithParameters(parameters: nil, filePathKey: "file", imageDataKey: imageData as NSData, boundary: boundary) as Data
        
        let task = sharedSession.uploadTask(with: request as URLRequest, from: dataToUpload) { (data, response, error) in
            
            if(error != nil){
                print("\(error!.localizedDescription)")
            }
            
            guard let responseData = data else {
                print("no response data")
                return
            }
            
            DispatchQueue.main.async {
                if let responseString = String(data: responseData, encoding: .utf8) {
                    print("uploaded to: \(responseString)")
                    self.myLabel.text = "successful"
                } else {
                    print("Nothong/////")
                    self.myLabel.text = "nopeee"
                }
            }
        }
        
        task.resume()

    }
    
    
    func createBodyWithParameters(parameters: [String: String]?, filePathKey: String?, imageDataKey: NSData, boundary: String) -> NSData {
        let body = NSMutableData();

        if parameters != nil {
            for (key, value) in parameters! {
                body.appendString(string: "--\(boundary)\r\n")
                body.appendString(string: "Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
                body.appendString(string: "\(value)\r\n")
            }
        }

        let filename = "user-profile.jpg"

        let mimetype = "image/jpg"

        body.appendString(string: "--\(boundary)\r\n")
        body.appendString(string: "Content-Disposition: form-data; name=\"\(filePathKey!)\"; filename=\"\(filename)\"\r\n")
        body.appendString(string: "Content-Type: \(mimetype)\r\n\r\n")
        body.append(imageDataKey as Data)
        body.appendString(string: "\r\n")

        body.appendString(string: "--\(boundary)--\r\n")

        return body
    }
    
    func generateBoundaryString() -> String {
        return "Boundary-\(NSUUID().uuidString)"
    }
}

extension NSMutableData {

    func appendString(string: String) {
        let data = string.data(using: String.Encoding.utf8, allowLossyConversion: true)
        append(data!)
    }
}

