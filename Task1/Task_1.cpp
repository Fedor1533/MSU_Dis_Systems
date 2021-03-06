#include <stdio.h>
#include <vector>
#include "mpi.h"
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <string.h>

namespace fs = std::filesystem;

std::string Text(int i, int length = 10)
{
    // Create text to write
    std::string text = std::string("");
    for (int k = 1; k < length; ++k)
    {
        text += std::to_string(i);
    }
    text += " \n";
    return text;
}

int main(int argc, char* argv[])
{

    // Current working path
    const fs::path workdir = fs::current_path();
    // path to all directories
    fs::path full_path = workdir / "file_sys";

    int rank, num_tasks;
    // Read quorum and right quorum for voting
    int N_r = 1, N_w = 11, all_servers = 11;
    int Ts = 100, Tb = 1, time = 0;
    int N = 10; // text length

    // number of processes - 12, 0 - main process, 1-11 - file servers
    MPI_Init(&argc, &argv);  // starts MPI 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks); // get number of processes
    MPI_Status status;
    MPI_Request request;

    if (rank == 0)
    {
        //this process attempts to do 3 write operations and 10 read operations
        int read = 1, write = 2, stop = 3;
        int confirm; // to confirm file version changes
        std::vector<int> process_version = { rank, 0 };

        // Try 3 write operations
        for (int i = 0; i < 3; ++i)
        {
            std::vector<std::vector<int>> processes_versions; // to collect N_w processes's id and it's versions
            processes_versions.resize(N_w);
            // Send write requests to Nw processes
            for (int j = 1; j <= N_w; ++j)
            {
                MPI_Send(&write, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                time += Ts;
                time += Tb * 4;
            }

            // Create text to write
            auto text = Text(i, N);
            // std::cout << "Text: " << text;

            //Send text to all N_w processes
            for (int p = 1; p <= N_w; ++p)
            {
                MPI_Send(text.c_str(), N + 1, MPI_CHAR, p, 2, MPI_COMM_WORLD);
                time += Ts;
                time += Tb * N;
            }

            // Receive file versions to confirm changes
            int version;
            for (int p = 1; p <= N_w; ++p)
            {
                MPI_Recv(&confirm, 1, MPI_INT, p, 2, MPI_COMM_WORLD, &status);
                if (p == 1) { version = confirm; }
                else if (version != confirm)
                {
                    printf("STOP: Received different version: %d from process: %d\n", confirm, p);
                    break;
                }
                time += Ts;
                time += Tb * 4;
            }
            // printf("Confirmed version %d\n", version);
        }

        // Try 10 read operations
        for (int i = 0; i < 10; ++i)
        {
            process_version = { rank, 0 };
            std::vector<std::vector<int>> processes_versions;
            processes_versions.resize(N_r);
            // Send read requests to Nr processes
            for (int j = 1; j <= N_r; ++j)
            {
                MPI_Send(&read, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                time += Ts;
                time += Tb * 4;
            }

            // Receive file versions and processes's ids from Nr processes
            for (int j = 1; j <= N_r; ++j)
            {
                MPI_Recv(process_version.data(), 2, MPI_INT, j, 1, MPI_COMM_WORLD, &status);
                time += Ts;
                time += Tb * 8;
                processes_versions[j - 1] = process_version;
            }

            // Find max version of file
            int max_version = 0, process_id = 1;
            for (auto proc_ver : processes_versions)
            {
                if (proc_ver[1] > max_version)
                {
                    process_id = proc_ver[0];
                    max_version = proc_ver[1];
                }
            }

            printf("Read version: %d from process: %d\n", max_version, process_id);
            fs::path full_path = workdir / "file_sys";
            fs::path file_path = full_path / ("test" + std::to_string(process_id)) / ("version" + std::to_string(max_version) + ".txt");
            // std::cout << "Read from file:" << file_path << std::endl;
            std::fstream file(file_path);

            int length = fs::file_size(file_path);
            char* buffer = new char[length + 1];
            buffer[length] = 0;
            // read data as a block:
            file.read(buffer, length);
            std::cout << "Read from file: " << buffer << std::endl;
            file.close();
        }

        std::cout << "\n Resulted time: " << time << std::endl;

        // Send Stop requests
        for (int p = 1; p <= all_servers; ++p)
        {
            MPI_Send(&stop, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        int req_type = 0; // type of request
        std::vector<int> process_version = { rank, 0 };
        int actual_version = 0;
        std::string server_dir = "test" + std::to_string(rank);

        std::fstream file;
        fs::path file_path;
        fs::path new_file_path;

        // creating "server directory" if not exist
        if (!fs::is_directory(workdir / "file_sys" / server_dir) || !fs::exists(workdir / "file_sys" / server_dir))
        {
            std::cout << "Create: " << "/file_sys/" + server_dir << std::endl;
            fs::create_directories(workdir / "file_sys" / server_dir);
            std::cout << "Copy " << "test_file.txt" << " to " << "file_sys/test/" + server_dir + "/version0.txt" << std::endl;
            fs::copy(workdir / "test_file.txt", workdir / "file_sys" / server_dir / "version0.txt");
        }
        
        while (true)
        {

            if (req_type == 1)
            {
                // Read request
                MPI_Send(process_version.data(), 2, MPI_INT, 0, 1, MPI_COMM_WORLD);
                req_type = 0;
            }
            else if (req_type == 2)
            {
                // Write request
                // Receive text to write
                char* buf = new char[N + 1];
                MPI_Recv(buf, N + 1, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
                std::string text(buf, N + 1);
                delete[] buf;

                // write text to new version of file
                file_path = full_path / ("test" + std::to_string(process_version[0])) / ("version" + std::to_string(process_version[1]) + ".txt");
                new_file_path = full_path / ("test" + std::to_string(process_version[0])) / ("version" + std::to_string(process_version[1] + 1) + ".txt");
                fs::copy(file_path, new_file_path);
                file.open(new_file_path, std::ios::app);
                file << text;
                file.close();

                process_version[1] += 1;
                // confirm that server changed it's file version
                MPI_Send(&process_version[1], 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                req_type = 0;
            }
            else
            {
                MPI_Recv(&req_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }

            if (req_type == 3)
            {
                // Stop request
                break;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
