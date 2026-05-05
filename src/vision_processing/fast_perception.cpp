#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <ros/serialization.h>
#include <ros/time.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/point_cloud.h>

namespace py = pybind11;

// 1. PointCloud2 Creator
py::bytes create_cloud_xyzrgb(py::array_t<float> points, int r, int g, int b, const std::string& frame_id) {
    auto buf = points.request();
    float* ptr = (float*)buf.ptr;
    int num_points = buf.shape[0];

    sensor_msgs::PointCloud2 msg;
    msg.header.frame_id = frame_id;
    // We intentionally do not set stamp here because ros::Time::now() requires ros::init() to have been called.
    // Python will set the timestamp after deserialization.
    
    msg.height = 1;
    msg.width = num_points;
    msg.is_bigendian = false;
    msg.is_dense = false;
    
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(num_points);
    
    sensor_msgs::PointCloud2Iterator<float> iter_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(msg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(msg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(msg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(msg, "b");
    
    for (int i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b) {
        *iter_x = ptr[i*3 + 0];
        *iter_y = ptr[i*3 + 1];
        *iter_z = ptr[i*3 + 2];
        *iter_r = r;
        *iter_g = g;
        *iter_b = b;
    }
    
    uint32_t serial_size = ros::serialization::serializationLength(msg);
    std::string buffer(serial_size, ' ');
    ros::serialization::OStream stream((uint8_t*)buffer.data(), serial_size);
    ros::serialization::serialize(stream, msg);
    
    return py::bytes(buffer);
}

// 2. Radius Outlier Removal using PCL
py::array_t<float> radius_outlier_removal(py::array_t<float> points, float radius, int min_neighbors) {
    auto buf = points.request();
    float* ptr = (float*)buf.ptr;
    int num_points = buf.shape[0];
    
    if (num_points == 0) {
        return py::array_t<float>(std::vector<ssize_t>{0, 3});
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = num_points;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    
    for (int i = 0; i < num_points; ++i) {
        cloud->points[i].x = ptr[i*3 + 0];
        cloud->points[i].y = ptr[i*3 + 1];
        cloud->points[i].z = ptr[i*3 + 2];
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(cloud);
    outrem.setRadiusSearch(radius);
    outrem.setMinNeighborsInRadius(min_neighbors);
    outrem.filter(*cloud_filtered);
    
    int out_points = cloud_filtered->points.size();
    py::array_t<float> result(std::vector<ssize_t>{out_points, 3});
    auto res_buf = result.request();
    float* res_ptr = (float*)res_buf.ptr;
    
    for (int i = 0; i < out_points; ++i) {
        res_ptr[i*3 + 0] = cloud_filtered->points[i].x;
        res_ptr[i*3 + 1] = cloud_filtered->points[i].y;
        res_ptr[i*3 + 2] = cloud_filtered->points[i].z;
    }
    
    return result;
}

PYBIND11_MODULE(fast_perception_module, m) {
    m.def("create_cloud_xyzrgb", &create_cloud_xyzrgb);
    m.def("radius_outlier_removal", &radius_outlier_removal);
}
